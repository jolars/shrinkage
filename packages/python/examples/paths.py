"""Fit, inspect, and predict Gaussian and logistic regularization paths.

Run from the repository with:
    uv run --locked --project packages/python python packages/python/examples/paths.py
"""

import numpy as np

from shrinkage import (
    Binomial,
    ChainFusedLasso,
    ClusteredLasso,
    Gaussian,
    Hybrid,
    PathOptions,
    Problem,
    ProximalGradient,
    Slope,
    StoppingCriterion,
)


def main() -> None:
    rng = np.random.default_rng(71)
    x = rng.normal(size=(30, 4))
    eta = 0.3 + x @ np.array([1.5, 1.5, -1.0, -1.0])
    probabilities = np.exp(-np.logaddexp(0, -eta))
    stopping = StoppingCriterion.proximal_gradient_mapping(1e-8)
    options = PathOptions(nlambda=5, lambda_min_ratio=0.05, early_stopping=False)
    for penalty in (
        ClusteredLasso(l1=1.0, fusion=0.3),
        ChainFusedLasso(l1=1.0, fusion=0.3, weights=[1, 0, 1]),
        Slope([1.0, 0.7, 0.3, 0.1]),
    ):
        for datafit in (Gaussian(eta), Binomial(probabilities)):
            problem = Problem(x, datafit, penalty)
            path = problem.fit_regularization_path_with(
                Hybrid(stopping=stopping), options
            )
            reference = problem.fit_regularization_path_with(
                ProximalGradient(stopping=stopping), options
            )
            assert path.termination == reference.termination == "completed"
            assert path.coefficients.shape == (5, 4)
            assert path.intercepts.shape == (5,)
            assert path.predict(x[:3]).shape == (3, 5)
            np.testing.assert_allclose(path.predict(x), reference.predict(x), atol=1e-5)
            for point, tuning in zip(path.fits, path.penalties, strict=True):
                assert point.termination == "converged"
                cold = Problem(x, datafit, tuning).fit_with(Hybrid(stopping=stopping))
                np.testing.assert_allclose(point.objective, cold.objective, atol=1e-9)
                assert point.working_set_diagnostics.policy.kind == "ever_active"
            explicit = problem.fit_path_with(
                Hybrid(stopping=stopping), reversed(path.penalties)
            )
            np.testing.assert_allclose(
                explicit.predict(x), path.predict(x)[:, ::-1], atol=1e-5
            )
            print(
                type(penalty).__name__,
                path.family,
                f"points={len(path)}",
                f"maximum={path.maximum_penalty:.6g}",
                f"last_objective={path.fits[-1].objective:.6g}",
            )


if __name__ == "__main__":
    main()
