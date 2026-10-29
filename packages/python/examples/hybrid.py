"""Fit all MVP penalties and families through the installed Python package.

Run from the repository with:
    uv run --locked --project packages/python python packages/python/examples/hybrid.py
"""

import numpy as np

from shrinkage import (
    Binomial,
    ChainFusedLasso,
    ClusteredLasso,
    Gaussian,
    Hybrid,
    Problem,
    ProximalGradient,
    Slope,
    StoppingCriterion,
)


def main() -> None:
    rng = np.random.default_rng(71)
    x = rng.normal(size=(30, 4))
    eta = 0.3 + x @ np.array([1.5, 1.5, -1.0, -1.0])
    probabilities = 1 / (1 + np.exp(-eta))
    stopping = StoppingCriterion.proximal_gradient_mapping(1e-8)
    for penalty in (
        ClusteredLasso(l1=0.02, fusion=0.01),
        ChainFusedLasso(l1=0.02, fusion=0.01, weights=[1, 0, 1]),
        Slope([0.05, 0.03, 0.02, 0.01]),
    ):
        for datafit in (Gaussian(eta), Binomial(probabilities)):
            problem = Problem(x, datafit, penalty, normalization="auto")
            hybrid = problem.fit_with(Hybrid(stopping=stopping))
            reference = problem.fit_with(ProximalGradient(stopping=stopping))
            assert hybrid.termination == reference.termination == "converged"
            np.testing.assert_allclose(hybrid.objective, reference.objective, atol=1e-9)
            np.testing.assert_allclose(
                hybrid.predict(x), reference.predict(x), atol=1e-5
            )
            predictions = hybrid.predict(x[:3])
            if hybrid.family == "binomial":
                # This form avoids overflow for logits with large magnitude.
                predictions = np.exp(-np.logaddexp(0, -predictions))
            print(
                type(penalty).__name__,
                hybrid.family,
                f"objective={hybrid.objective:.6g}",
                f"clusters={len(hybrid.cluster_values)}",
                f"predictions={predictions}",
            )


if __name__ == "__main__":
    main()
