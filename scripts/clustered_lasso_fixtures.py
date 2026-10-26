# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "cvxpy==1.9.3", "clarabel==0.11.1"]
# ///
"""Generate independent all-pairs epigraph QP fixtures without Shrinkage."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import cvxpy as cp
import numpy as np

REVISION = "293760ea4040351a654906add5d7ca855be1deb9"
OPTIONS = {"tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11, "tol_feas": 1e-11}


def cases():
    """Retain the research oracle gate's seed and add free-intercept cases."""
    rng = np.random.default_rng(20261014)
    x = rng.normal(size=(12, 5))
    x -= x.mean(axis=0)
    x[:, 1] = 0.8 * x[:, 0] + 0.6 * x[:, 1]
    y = x @ np.array([-1.5, -1.5, 0.0, 0.8, 1.2]) + rng.normal(0, 0.1, 12)
    y -= y.mean()
    for name, l1, fusion in [
        ("correlated", 0.2, 0.15),
        ("lasso_limit", 0.2, 0.0),
        ("fusion_limit", 0.0, 0.15),
        ("least_squares", 0.0, 0.0),
        ("all_zero", 100.0, 0.15),
        ("complete_fusion", 0.2, 100.0),
    ]:
        yield name, x, y, l1, fusion, False
    wide = rng.normal(size=(5, 7))
    wide -= wide.mean(axis=0)
    response = rng.normal(size=5)
    response -= response.mean()
    yield "underdetermined", wide, response, 0.1, 0.07, False
    singular = x.copy()
    singular[:, 1] = singular[:, 0]
    singular[:, 4] = 0
    yield "duplicate_and_zero_columns", singular, y, 0.2, 0.15, False
    yield "nonunique_least_squares", singular, y, 0.0, 0.0, False
    yield "zero_design", np.zeros((3, 4)), np.array([1.0, -2.0, 1.0]), 0.2, 0.1, False
    yield (
        "analytic_diagonal_lasso",
        np.diag([1.0, 2.0, 3.0]),
        np.array([-2.0, 1.0, 0.05]),
        0.3,
        0.0,
        False,
    )
    yield (
        "analytic_diagonal_both",
        np.diag([1.0, 2.0, 3.0]),
        np.array([-1.5, 1.05, 19.0 / 6.0]),
        0.1,
        0.2,
        False,
    )
    yield "free_intercept", x + 1.5, y + 2.0, 0.2, 0.15, True
    yield "singular_free_intercept", singular + 0.5, y + 3.0, 0.2, 0.15, True


def solve(name, x, y, research_l1, research_fusion, fit_intercept):
    """Use epigraph inequalities and their duals, with no sorting or prox."""
    n, p = x.shape
    l1, fusion = research_l1 / n, research_fusion / n
    beta = cp.Variable(p)
    intercept = cp.Variable() if fit_intercept else 0.0
    constraints, terms = [], []
    l1_constraints = fusion_constraints = None
    if l1:
        magnitude = cp.Variable(p, nonneg=True)
        l1_constraints = [magnitude >= beta, magnitude >= -beta]
        constraints.extend(l1_constraints)
        terms.append(l1 * cp.sum(magnitude))
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    if fusion:
        differences = cp.hstack([beta[i] - beta[j] for i, j in pairs])
        distance = cp.Variable(len(pairs), nonneg=True)
        fusion_constraints = [distance >= differences, distance >= -differences]
        constraints.extend(fusion_constraints)
        terms.append(fusion * cp.sum(distance))
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x @ beta + intercept - y) / (2 * n) + sum(terms)),
        constraints,
    )
    problem.solve(solver="CLARABEL", **OPTIONS)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"{name}: oracle status {problem.status}")
    point = np.asarray(beta.value)
    offset = float(intercept.value) if fit_intercept else 0.0
    fitted = x @ point + offset
    objective = float(
        np.sum((fitted - y) ** 2) / (2 * n)
        + l1 * sum(abs(point))
        + fusion * sum(abs(point[i] - point[j]) for i, j in pairs)
    )
    l1_dual = (
        np.zeros(p)
        if l1_constraints is None
        else l1_constraints[0].dual_value - l1_constraints[1].dual_value
    )
    fusion_dual = (
        np.zeros(len(pairs))
        if fusion_constraints is None
        else fusion_constraints[0].dual_value - fusion_constraints[1].dual_value
    )
    subgradient = l1_dual.copy()
    for (i, j), value in zip(pairs, fusion_dual, strict=True):
        subgradient[i] += value
        subgradient[j] -= value
    stationarity = np.max(np.abs(x.T @ (fitted - y) / n + subgradient))
    if stationarity > 1e-8 or abs(objective - problem.value) > 1e-8 * (
        1 + abs(objective)
    ):
        raise RuntimeError(f"{name}: independent objective or KKT check failed")
    return {
        "name": name,
        "design": x.tolist(),
        "response": y.tolist(),
        "l1": l1,
        "fusion": fusion,
        "fit_intercept": fit_intercept,
        "coefficients": point.tolist(),
        "intercept": offset,
        "fitted": fitted.tolist(),
        "objective": objective,
        "l1_dual": l1_dual.tolist(),
        "fusion_dual": fusion_dual.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/clustered_lasso/reference.json",
    )
    options = parser.parse_args()
    output = {
        "provenance": {
            "research_revision": REVISION,
            "seed": 20261014,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {
                name: version(name) for name in ["numpy", "cvxpy", "clarabel"]
            },
            "oracle": "explicit unordered-pair epigraph QP, averaged squared loss",
            "solver_options": OPTIONS,
        },
        "cases": [solve(*case) for case in cases()],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
