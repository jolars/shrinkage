# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "cvxpy==1.9.3", "clarabel==0.11.1"]
# ///
"""Generate independent weighted-chain epigraph QP fixtures without Shrinkage."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import cvxpy as cp
import numpy as np

OPTIONS = {"tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11, "tol_feas": 1e-11}


def solve(name, x, y, l1, fusion, weights, fit_intercept=False, step=None):
    """Use explicit inequalities and their duals, independently of any prox."""
    n, p = x.shape
    beta = cp.Variable(p)
    intercept = cp.Variable() if fit_intercept else 0.0
    magnitude = cp.Variable(p, nonneg=True)
    distance = cp.Variable(p - 1, nonneg=True)
    differences = beta[1:] - beta[:-1]
    constraints = [
        magnitude >= beta,
        magnitude >= -beta,
        distance >= differences,
        distance >= -differences,
    ]
    divisor = n if step is None else step
    problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(x @ beta + intercept - y) / (2 * divisor)
            + l1 * cp.sum(magnitude)
            + fusion * (weights @ distance)
        ),
        constraints,
    )
    problem.solve(solver="CLARABEL", **OPTIONS)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"{name}: oracle status {problem.status}")
    point = np.asarray(beta.value)
    offset = float(intercept.value) if fit_intercept else 0.0
    fitted = x @ point + offset
    objective = float(
        np.sum((fitted - y) ** 2) / (2 * divisor)
        + l1 * np.abs(point).sum()
        + fusion * (weights @ np.abs(np.diff(point)))
    )
    l1_dual = constraints[0].dual_value - constraints[1].dual_value
    fusion_dual = constraints[2].dual_value - constraints[3].dual_value
    subgradient = l1_dual.copy()
    subgradient[:-1] -= fusion_dual
    subgradient[1:] += fusion_dual
    stationarity = np.max(np.abs(x.T @ (fitted - y) / divisor + subgradient))
    if fit_intercept:
        stationarity = max(stationarity, abs(np.sum(fitted - y) / divisor))
    if stationarity > 1e-8 or abs(objective - problem.value) > 1e-8 * (
        1 + abs(objective)
    ):
        raise RuntimeError(f"{name}: independent objective or KKT check failed")
    result = {
        "name": name,
        "l1": l1,
        "fusion": fusion,
        "weights": weights.tolist(),
        "coefficients": point.tolist(),
        "objective": objective,
        "l1_dual": l1_dual.tolist(),
        "fusion_dual": fusion_dual.tolist(),
    }
    if step is None:
        result.update(
            design=x.tolist(),
            response=y.tolist(),
            fit_intercept=fit_intercept,
            intercept=offset,
            fitted=fitted.tolist(),
        )
    else:
        result.update(input=y.tolist(), step=step)
    return result


def fits():
    rng = np.random.default_rng(20261026)
    x = rng.normal(size=(16, 6))
    x[:, 1] = 0.8 * x[:, 0] + 0.6 * x[:, 1]
    y = x @ np.array([-1.5, -1.5, 0.0, 0.8, 0.8, -1.5])
    y += rng.normal(0, 0.1, 16)
    weights = np.array([0.25, 2.0, 0.0, 1.5, 0.75])
    for name, l1, fusion in [
        ("correlated", 0.03, 0.12),
        ("lasso_limit", 0.03, 0.0),
        ("fusion_limit", 0.0, 0.12),
        ("least_squares", 0.0, 0.0),
        ("all_zero", 100.0, 0.12),
        ("strong_fusion_with_cut", 0.03, 100.0),
    ]:
        yield solve(name, x, y, l1, fusion, weights)
    yield solve("uniform", x, y, 0.03, 0.12, np.ones(5))
    yield solve("all_edges_cut", x, y, 0.03, 0.12, np.zeros(5))
    yield solve("complete_fusion", x, y, 0.03, 100.0, np.ones(5))
    yield solve("free_intercept", x + 1.5, y + 2.0, 0.03, 0.12, weights, True)
    singular = x.copy()
    singular[:, 1] = singular[:, 0]
    singular[:, 4] = 0.0
    yield solve("duplicate_and_zero_columns", singular, y, 0.03, 0.12, weights)
    yield solve("nonunique_least_squares", singular, y, 0.0, 0.0, weights)
    yield solve(
        "singular_free_intercept", singular + 0.5, y + 3.0, 0.03, 0.12, weights, True
    )
    wide = rng.normal(size=(5, 7))
    yield solve("underdetermined", wide, rng.normal(size=5), 0.1, 0.07, np.ones(6))
    yield solve(
        "zero_design",
        np.zeros((3, 4)),
        np.array([1.0, -2.0, 1.0]),
        0.2,
        0.1,
        np.ones(3),
    )


def proximals():
    for name, values, weights in [
        ("repeated_separated", [-2.0, -2.0, 1.0, 0.0, -2.0], [1.0, 0.2, 2.0, 0.5]),
        ("cuts", [3.0, -1.0, -1.0, 2.0, 0.1], [2.0, 0.0, 0.25, 1.0]),
        ("oscillating", [2.0, -3.0, 1.0, -2.0, 4.0], [0.25, 3.0, 0.125, 2.0]),
    ]:
        values = np.asarray(values)
        for l1, fusion in [(0.2, 0.7), (0.0, 0.7), (0.2, 0.0), (0.0, 0.0)]:
            for step in [0.25, 2.0]:
                yield solve(
                    f"{name}_{l1}_{fusion}_{step}",
                    np.eye(len(values)),
                    values,
                    l1,
                    fusion,
                    np.asarray(weights),
                    step=step,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/chain_fused_lasso/reference.json",
    )
    options = parser.parse_args()
    output = {
        "provenance": {
            "seed": 20261026,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {
                name: version(name) for name in ["numpy", "cvxpy", "clarabel"]
            },
            "oracle": "explicit weighted-chain epigraph QP",
            "solver_options": OPTIONS,
        },
        "proximal": list(proximals()),
        "fits": list(fits()),
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
