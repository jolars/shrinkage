# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "cvxpy==1.9.3", "clarabel==0.11.1"]
# ///
"""Generate independent SLOPE Gaussian fixtures through top-k epigraph QPs."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import cvxpy as cp
import numpy as np

OPTIONS = {"tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11, "tol_feas": 1e-11}


def solve(name, x, y, weights, fit_intercept=False):
    """Express sorted L1 through sums of the largest absolute coefficients."""
    n, p = x.shape
    beta = cp.Variable(p)
    intercept = cp.Variable() if fit_intercept else 0.0
    magnitude = cp.Variable(p, nonneg=True)
    constraints = [magnitude >= beta, magnitude >= -beta]
    regularizer = weights[-1] * cp.sum(magnitude)
    for k in range(1, p):
        difference = weights[k - 1] - weights[k]
        if difference > 0:
            regularizer += difference * cp.sum_largest(magnitude, k)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x @ beta + intercept - y) / (2 * n) + regularizer),
        constraints,
    )
    problem.solve(solver="CLARABEL", **OPTIONS)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"{name}: oracle status {problem.status}")
    point = np.asarray(beta.value)
    offset = float(intercept.value) if fit_intercept else 0.0
    fitted = x @ point + offset
    residual = fitted - y
    subgradient = constraints[0].dual_value - constraints[1].dual_value
    penalty = weights @ np.sort(np.abs(point))[::-1]
    objective = float(np.sum(residual**2) / (2 * n) + penalty)
    stationarity = np.max(np.abs(x.T @ residual / n + subgradient))
    if fit_intercept:
        stationarity = max(stationarity, abs(np.mean(residual)))
    feasibility = np.max(
        np.cumsum(np.sort(np.abs(subgradient))[::-1]) - np.cumsum(weights)
    )
    support_gap = abs(point @ subgradient - penalty)
    if max(stationarity, feasibility, support_gap) > 1e-8:
        raise RuntimeError(f"{name}: independent KKT certificate failed")
    if abs(objective - problem.value) > 1e-8 * (1 + abs(objective)):
        raise RuntimeError(f"{name}: independent objective check failed")
    return {
        "name": name,
        "design": x.tolist(),
        "response": y.tolist(),
        "weights": weights.tolist(),
        "fit_intercept": fit_intercept,
        "coefficients": point.tolist(),
        "intercept": offset,
        "fitted": fitted.tolist(),
        "objective": objective,
        "subgradient": subgradient.tolist(),
    }


def fits():
    rng = np.random.default_rng(20261026)
    x = rng.normal(size=(16, 6))
    x[:, 1] = 0.8 * x[:, 0] + 0.6 * x[:, 1]
    y = x @ np.array([-1.5, 1.5, 0.0, 0.8, -0.8, -1.5])
    y += rng.normal(0, 0.1, 16)
    weights = np.array([0.3, 0.2, 0.15, 0.1, 0.05, 0.02])
    for name, profile in [
        ("correlated", weights),
        ("repeated_and_zero", np.array([0.3, 0.3, 0.1, 0.1, 0.0, 0.0])),
        ("lasso", np.full(6, 0.1)),
        ("least_squares", np.zeros(6)),
        ("all_zero", np.full(6, 100.0)),
        ("one_magnitude", np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]:
        yield solve(name, x, y, profile)
    yield solve("free_intercept", x + 1.5, y + 2.0, weights, True)
    singular = x.copy()
    singular[:, 1] = singular[:, 0]
    singular[:, 4] = 0.0
    yield solve("duplicate_and_zero_columns", singular, y, weights)
    yield solve("nonunique_least_squares", singular, y, np.zeros(6))
    yield solve("singular_free_intercept", singular + 0.5, y + 3.0, weights, True)
    yield solve(
        "underdetermined",
        rng.normal(size=(5, 7)),
        rng.normal(size=5),
        np.linspace(0.2, 0.01, 7),
    )
    yield solve(
        "zero_design",
        np.zeros((3, 4)),
        np.array([1.0, -2.0, 1.0]),
        np.array([0.3, 0.2, 0.1, 0.0]),
    )
    yield solve(
        "opposite_signs",
        2.0 * np.eye(2),
        np.array([-3.0, 3.0]),
        np.array([3.0, 1.0]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/slope/reference.json",
    )
    options = parser.parse_args()
    output = {
        "provenance": {
            "seed": 20261026,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {
                name: version(name) for name in ["numpy", "cvxpy", "clarabel"]
            },
            "oracle": "top-k magnitude epigraph QP",
            "solver_options": OPTIONS,
        },
        "fits": list(fits()),
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
