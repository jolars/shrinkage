"""Metewand workers for one dense, single-alpha Python lasso comparison."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

# Numerical libraries read these limits when they are imported.
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
from metewand import (  # noqa: E402
    Artifact,
    Dataset,
    Evaluation,
    read_json,
    serve_dataset_materializer,
    serve_evaluator,
    serve_implementation,
)


def materialize(context):
    rng = np.random.default_rng(context.seed)
    n, p = context.parameters["n_samples"], context.parameters["n_features"]
    x = np.asfortranarray(rng.standard_normal((n, p)))
    beta = np.zeros(p)
    beta[:20] = np.tile([1.0, -1.0], 10)
    y = x @ beta + rng.standard_normal(n)
    x -= x.mean(axis=0)
    y -= y.mean()
    alpha = context.parameters["alpha_fraction"] * np.max(np.abs(x.T @ y)) / n
    np.savez(context.output_dir / "data.npz", x=x, y=y, alpha=alpha)
    return Dataset(files=[Artifact("data.npz", "application/octet-stream")])


def load_data(directory: Path):
    archive = directory / "data.npz"
    if archive.exists():
        with np.load(archive, allow_pickle=False) as data:
            return data["x"], data["y"], float(data["alpha"])
    # Analytic reference cases use small, inline JSON datasets.
    data = read_json(directory / "data.json")
    return np.asarray(data["x"], dtype=float), np.asarray(data["y"]), data["alpha"]


def evaluate_solution(x, y, alpha, beta, target=1e-8):
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (x.shape[1],) or not np.all(np.isfinite(beta)):
        return Evaluation({}, "invalid")
    n = len(y)
    residual = y - x @ beta
    gradient = x.T @ residual / n
    objective = residual @ residual / (2 * n) + alpha * np.abs(beta).sum()
    dual = residual / n
    dual *= min(1.0, alpha / max(float(np.max(np.abs(gradient))), alpha))
    dual_objective = y @ dual - n * (dual @ dual) / 2
    reference_loss = y @ y / (2 * n)
    gap = max(0.0, float(objective - dual_objective))
    violation = np.where(
        beta != 0,
        np.abs(gradient - alpha * np.sign(beta)),
        np.maximum(np.abs(gradient) - alpha, 0),
    )
    metrics = {
        "objective": float(objective),
        "duality_gap": gap,
        "relative_gap": float(gap / reference_loss),
        "kkt_violation": float(np.max(violation)),
        "alpha": float(alpha),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        return Evaluation({}, "invalid")
    return Evaluation(
        metrics, "reached" if metrics["relative_gap"] <= target else "not_reached"
    )


def evaluate(context):
    x, y, alpha = load_data(context.dataset_dir)
    result = read_json(context.result_dir / "result.json")["data"]
    target = context.semantics["one_shot_completion"]["relative_gap"]
    return evaluate_solution(x, y, alpha, result["coefficients"], target)


class LassoWorker:
    def __init__(self, implementation):
        self.implementation = implementation
        self.version = version(implementation)
        self.software = {
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("numpy", "scipy", "metewand")
            },
        }
        if implementation == "shrinkage":
            from shrinkage import Lasso, StoppingCriterion, _shrinkage

            self.model = Lasso
            self.stopping = StoppingCriterion.duality_gap(relative=1e-8)
            self.software["native_extension_sha256"] = hashlib.sha256(
                Path(_shrinkage.__file__).read_bytes()
            ).hexdigest()
        else:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import Lasso

            # Truncated fits supply the intermediate convergence points.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model = Lasso

    def prepare(self, context):
        return load_data(context.dataset_dir)

    def execute(self, state, context):
        x, y, alpha = state
        iterations = context.observation_control["iterations"]
        if self.implementation == "shrinkage":
            return self.model(
                alpha,
                fit_intercept=False,
                normalization="none",
                stopping=self.stopping,
                max_iterations=iterations,
            ).fit(x, y)
        # sklearn scales its gap threshold by ||y||^2/n, twice the null loss.
        return self.model(
            alpha=alpha,
            fit_intercept=False,
            tol=5e-9,
            max_iter=iterations,
            selection="cyclic",
            precompute=False,
            copy_X=True,
            warm_start=False,
        ).fit(x, y)

    def result(self, output, context):
        coefficients = (
            output.coefficients if self.implementation == "shrinkage" else output.coef_
        )
        return {"coefficients": coefficients.tolist()}

    def statistics(self, output, context):
        from threadpoolctl import threadpool_info

        return {
            "implementation": self.implementation,
            "version": self.version,
            "software": self.software,
            "iterations": int(
                output.iterations
                if self.implementation == "shrinkage"
                else output.n_iter_
            ),
            "iteration_limit": context.observation_control["iterations"],
            "threadpools": threadpool_info(),
        }

    def reset(self, state, context):
        pass


if __name__ == "__main__":
    role = sys.argv[1]
    if role == "dataset":
        serve_dataset_materializer(materialize)
    elif role == "evaluate":
        serve_evaluator(evaluate)
    else:
        worker = LassoWorker(role)
        serve_implementation(
            prepare=worker.prepare,
            execute=worker.execute,
            result=worker.result,
            statistics=worker.statistics,
            reset=worker.reset,
        )
