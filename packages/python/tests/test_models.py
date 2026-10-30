"""Exercise composition and numerical results through the installed package."""

from __future__ import annotations

import gc
import json
import math
import sys
import threading
import unittest
import weakref
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt

from shrinkage import (
    Binomial,
    ChainFusedLasso,
    ClusteredLasso,
    Gaussian,
    Hybrid,
    HybridFit,
    HybridStep,
    Lasso,
    Normalization,
    Problem,
    ProximalFit,
    ProximalGradient,
    ScreeningDiagnostics,
    Slope,
    StoppingCriterion,
    WorkingSetDiagnostics,
    WorkingSetPolicy,
)

FloatArray = npt.NDArray[np.float64]
Penalty = ClusteredLasso | ChainFusedLasso | Slope
T = TypeVar("T")


def penalties() -> tuple[Penalty, ...]:
    return (
        ClusteredLasso(0.03, 0.02),
        ChainFusedLasso(0.03, 0.02, weights=[0.0, 2.0, 0.5]),
        Slope([0.08, 0.04, 0.02, 0.0]),
    )


def objective(
    fit: ProximalFit, x: FloatArray, y: FloatArray, penalty: Penalty
) -> float:
    eta = x @ fit.coefficients + fit.intercept
    if fit.family == "gaussian":
        loss = float(np.mean((y - eta) ** 2) / 2)
    else:
        loss = float(np.mean(np.logaddexp(0, eta) - y * eta))
    scales = fit.preprocessing.scales
    beta = fit.coefficients * (1 if scales is None else scales)
    if isinstance(penalty, Slope):
        return loss + float(np.asarray(penalty.weights) @ np.sort(abs(beta))[::-1])
    value = penalty.l1 * float(abs(beta).sum())
    if isinstance(penalty, ClusteredLasso):
        value += penalty.fusion * sum(
            abs(float(beta[i] - beta[j]))
            for i in range(len(beta))
            for j in range(i + 1, len(beta))
        )
    else:
        weights = 1 if penalty.weights is None else np.asarray(penalty.weights)
        value += penalty.fusion * float(np.sum(weights * abs(np.diff(beta))))
    return loss + value


def fit_problem(problem: Problem, solver: Hybrid | ProximalGradient) -> ProximalFit:
    return problem.fit_with(solver)


def zero_arrays(*arrays: FloatArray) -> None:
    for array in arrays:
        array.fill(0)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [0.0, 1.0, 3.0, 0.0],
                [1.0, 0.0, 3.0, 2.0],
                [2.0, 2.0, 3.0, 0.0],
                [3.0, -1.0, 3.0, 1.0],
                [4.0, 3.0, 3.0, 0.0],
                [5.0, 1.0, 3.0, 2.0],
                [6.0, 4.0, 3.0, 1.0],
                [7.0, -2.0, 3.0, 0.0],
            ]
        )
        self.y = np.array([1.0, 2.3, 0.8, 3.9, 4.2, 2.7, 6.1, 5.4])
        self.binary_y = np.array([0.1, 0.2, 0.25, 0.3, 0.65, 0.7, 0.8, 0.9])

    def test_all_diagnostics_match_typed_public_rust_fixtures(self) -> None:
        fixture = json.loads(
            Path(__file__).with_name("fixtures").joinpath("model_rust.json").read_text()
        )
        x = np.asarray(fixture["x"], dtype=np.float64)
        predict_x = np.asarray(fixture["predict_x"], dtype=np.float64)
        combinations = set()
        for case in fixture["cases"]:
            with self.subTest(case={k: v for k, v in case.items() if k != "expected"}):
                options = case["solver_options"].copy()
                options["stopping"] = StoppingCriterion(**options["stopping"])
                if case["solver"] == "hybrid":
                    options["working_set_policy"] = WorkingSetPolicy(
                        **options["working_set_policy"]
                    )
                    options["step_policy"] = HybridStep(**options["step_policy"])
                    solver: Hybrid | ProximalGradient = Hybrid(**options)
                else:
                    solver = ProximalGradient(**options)
                y = np.asarray(fixture[f"{case['family']}_y"], dtype=np.float64)
                datafit = (
                    Gaussian(y, fit_intercept=case["fit_intercept"])
                    if case["family"] == "gaussian"
                    else Binomial(y, fit_intercept=case["fit_intercept"])
                )
                penalty = penalties()[case["penalty"]]
                fit = Problem(x, datafit, penalty, normalization=case["normalization"])
                result = fit.fit_with(solver)
                expected = case["expected"]
                self.assertEqual(result.solver, case["solver"])
                self.assertEqual(result.family, case["family"])
                combinations.add((result.solver, result.family, case["penalty"]))
                for name, value in expected.items():
                    if name in ("centers", "scales"):
                        actual = getattr(result.preprocessing, name)
                    elif name == "predictions":
                        actual = result.predict(predict_x)
                    elif name == "screening_diagnostics":
                        self.assertEqual(
                            result.screening_diagnostics, ScreeningDiagnostics(**value)
                        )
                        continue
                    elif name == "working_set_diagnostics":
                        value = value.copy()
                        value["policy"] = WorkingSetPolicy(**value["policy"])
                        assert isinstance(result, HybridFit)
                        self.assertEqual(
                            result.working_set_diagnostics,
                            WorkingSetDiagnostics(**value),
                        )
                        continue
                    elif name == "step_policy":
                        assert isinstance(result, HybridFit)
                        self.assertEqual(result.step_policy, HybridStep(**value))
                        continue
                    else:
                        actual = getattr(result, name)
                    if value is None or isinstance(value, str | int):
                        self.assertEqual(actual, value, name)
                    else:
                        np.testing.assert_allclose(
                            actual, value, rtol=1e-10, atol=1e-12, err_msg=name
                        )
                self.assertEqual(result.stopping, options["stopping"])
                self.assertAlmostEqual(
                    result.objective, objective(result, x, y, penalty), delta=1e-11
                )
                if result.termination == "converged":
                    self.assertLessEqual(
                        result.stopping_value, result.stopping_threshold
                    )
        self.assertEqual(len(combinations), 12)

    def test_orthogonal_gaussian_solutions_and_exact_cluster_meaning(self) -> None:
        cases: tuple[tuple[Penalty, list[float], list[float]], ...] = (
            (
                ClusteredLasso(0.1, 0.2),
                [-2.2, -2.0, 2.0, 2.2],
                [-1.6, -1.6, 1.6, 1.6],
            ),
            (
                ChainFusedLasso(0.1, 0.4),
                [-2.2, -2.0, 2.0, 2.2],
                [-1.8, -1.8, 1.8, 1.8],
            ),
            (
                Slope([0.8, 0.4, 0.2, 0.0]),
                [-3.0, 3.0, 1.0, -1.0],
                [-2.4, 2.4, 0.9, -0.9],
            ),
            (
                ChainFusedLasso(0.1, 0.4, weights=[0.0, 2.0]),
                [2.0, -2.0, -2.0],
                [1.9, -1.9, -1.9],
            ),
            (
                ChainFusedLasso(0.1, 0.4, weights=[0.0, 1.0]),
                [2.0, 2.0, 2.0],
                [1.9, 1.9, 1.9],
            ),
        )
        stopping = StoppingCriterion.proximal_gradient_mapping(1e-10)
        for penalty, signal, expected in cases:
            x = np.sqrt(len(signal)) * np.eye(len(signal))
            y = x @ np.asarray(signal)
            for solver in (
                Hybrid(stopping=stopping),
                ProximalGradient(stopping=stopping),
            ):
                with self.subTest(penalty=penalty, solver=solver):
                    fit = Problem(
                        x,
                        Gaussian(y, fit_intercept=False),
                        penalty,
                        normalization="none",
                    ).fit_with(solver)
                    self.assertEqual(fit.termination, "converged")
                    np.testing.assert_allclose(fit.coefficients, expected, atol=2e-10)
                    self.assertAlmostEqual(
                        fit.objective, objective(fit, x, y, penalty), delta=1e-12
                    )
                    if isinstance(fit, HybridFit):
                        values = fit.cluster_values[fit.cluster_labels]
                        if isinstance(penalty, Slope):
                            values *= fit.coefficient_signs
                            self.assertEqual(
                                fit.cluster_labels[0], fit.cluster_labels[1]
                            )
                        np.testing.assert_allclose(values, expected, atol=2e-10)
                        if isinstance(penalty, ChainFusedLasso) and penalty.weights:
                            self.assertNotEqual(
                                fit.cluster_labels[0], fit.cluster_labels[1]
                            )

    def test_closed_form_logistic_fit_and_logits(self) -> None:
        x, y = np.ones((4, 1)), np.array([0.8, 0.7, 0.6, 0.9])
        for penalty in (
            ClusteredLasso(0.1, 2.0),
            ChainFusedLasso(0.1, 2.0),
            Slope([0.1]),
        ):
            for solver in (
                Hybrid(stopping=StoppingCriterion.proximal_gradient_mapping(1e-10)),
                ProximalGradient(
                    stopping=StoppingCriterion.proximal_gradient_mapping(1e-10)
                ),
            ):
                with self.subTest(penalty=penalty, solver=solver):
                    fit = Problem(
                        x,
                        Binomial(y, fit_intercept=False),
                        penalty,
                        normalization="none",
                    ).fit_with(solver)
                    self.assertEqual(fit.termination, "converged")
                    np.testing.assert_allclose(
                        fit.coefficients, [math.log(0.65 / 0.35)], atol=1e-9
                    )
                    np.testing.assert_allclose(fit.predict(x), x @ fit.coefficients)
                    gradient = 1 / (1 + np.exp(-fit.predict(x))) - y
                    self.assertLess(abs(float(gradient.mean()) + 0.1), 2e-10)

    def test_original_scale_prediction_and_cluster_reconstruction(self) -> None:
        for penalty in penalties():
            for datafit in (Gaussian(self.y), Binomial(self.binary_y)):
                fit = Problem(self.x, datafit, penalty).fit_with(Hybrid())
                scales = fit.preprocessing.scales
                assert scales is not None
                values = fit.cluster_values[fit.cluster_labels]
                if isinstance(penalty, Slope):
                    values *= fit.coefficient_signs
                np.testing.assert_allclose(
                    values, fit.coefficients * scales, atol=1e-12
                )
                new_x = self.x[:3] + 10
                np.testing.assert_allclose(
                    fit.predict(new_x), new_x @ fit.coefficients + fit.intercept
                )
                self.assertEqual(fit.cluster_labels.dtype, np.dtype(np.intp))
                self.assertEqual(fit.iterations, fit.pg_steps + fit.coordinate_sweeps)

    def test_zero_strength_limits(self) -> None:
        x = 2 * np.eye(4)
        for datafit in (
            Gaussian(np.array([-2.0, 1.0, 3.0, -1.0]), fit_intercept=False),
            Binomial(np.array([0.2, 0.4, 0.7, 0.8]), fit_intercept=False),
        ):
            y = cast(FloatArray, datafit.y)
            expected = (
                y / 2 if isinstance(datafit, Gaussian) else np.log(y / (1 - y)) / 2
            )
            for penalty in (
                ClusteredLasso(0.0, 0.0),
                ChainFusedLasso(0.0, 0.0),
                Slope([0.0] * 4),
            ):
                for solver in (Hybrid(), ProximalGradient()):
                    fit = Problem(x, datafit, penalty, normalization="none").fit_with(
                        solver
                    )
                    self.assertEqual(fit.termination, "converged")
                    np.testing.assert_allclose(fit.coefficients, expected, atol=1e-5)

    def test_screening_and_working_sets_preserve_the_original_solution(self) -> None:
        rng = np.random.default_rng(82)
        x = rng.normal(size=(40, 16))
        eta = x[:, :4] @ np.array([2.0, 2.0, -1.0, -1.0])
        for datafit in (Gaussian(eta), Binomial(1 / (1 + np.exp(-eta)))):
            for penalty in (
                ClusteredLasso(0.05, 0.01),
                ChainFusedLasso(0.05, 0.01),
                Slope(np.linspace(0.1, 0.01, 16)),
            ):
                problem = Problem(x, datafit, penalty)
                reference = problem.fit_with(
                    Hybrid(
                        screening=False,
                        working_set_policy=WorkingSetPolicy.disabled(),
                        stopping=StoppingCriterion.proximal_gradient_mapping(1e-8),
                    )
                )
                self.assertIsNone(reference.dual_certificate)
                self.assertIsNone(reference.dual_objective)
                self.assertIsNone(reference.duality_gap)
                for policy in (
                    WorkingSetPolicy.auto(),
                    WorkingSetPolicy.ever_active(),
                    WorkingSetPolicy.adaptive(1),
                    WorkingSetPolicy.adaptive(2, prune=False),
                ):
                    with self.subTest(
                        datafit=type(datafit), penalty=penalty, policy=policy
                    ):
                        fit = problem.fit_with(
                            Hybrid(
                                working_set_policy=policy,
                                stopping=StoppingCriterion.proximal_gradient_mapping(
                                    1e-8
                                ),
                            )
                        )
                        self.assertEqual(fit.termination, "converged")
                        self.assertAlmostEqual(
                            fit.objective, reference.objective, delta=1e-7
                        )
                        np.testing.assert_allclose(
                            fit.predict(x), reference.predict(x), atol=2e-5
                        )
                        self.assertNotEqual(
                            fit.working_set_diagnostics.policy.kind, "auto"
                        )

    def test_configuration_weights_are_owned_and_problem_inputs_are_live_until_fit(
        self,
    ) -> None:
        weights = np.array([0.08, 0.04, 0.02, 0.0])
        penalty = Slope(weights)
        weights[:] = -1
        self.assertEqual(penalty.weights, (0.08, 0.04, 0.02, 0.0))
        edge_weights = np.array([0.0, 2.0, 0.5])
        chain = ChainFusedLasso(0.03, 0.02, weights=edge_weights)
        edge_weights[:] = -1
        self.assertEqual(chain.weights, (0.0, 2.0, 0.5))
        y = self.y.copy()
        problem = Problem(self.x, Gaussian(y), penalty)
        first = problem.fit_with(Hybrid())
        y += 5
        second = problem.fit_with(Hybrid())
        np.testing.assert_allclose(
            second.predict(self.x), first.predict(self.x) + 5, atol=1e-5
        )

    def test_owned_fits_outlive_sources_and_return_independent_arrays(self) -> None:
        for solver in (Hybrid(), ProximalGradient()):
            x, y = self.x.copy(), self.y.copy()
            references = [weakref.ref(x), weakref.ref(y)]
            problem = Problem(x, Gaussian(y), penalties()[0])
            fit = problem.fit_with(solver)
            expected = fit.predict(x)
            x[:] = y[:] = -100
            del x, y, problem
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            arrays = [fit.coefficients, fit.predict(self.x), fit.dual_certificate]
            arrays += [fit.preprocessing.centers, fit.preprocessing.scales]
            if isinstance(fit, HybridFit):
                arrays += [fit.cluster_values, fit.coefficient_signs]
                fit.cluster_labels[:] = 100
            for array in arrays:
                if array is not None:
                    array[:] = 1000
            np.testing.assert_allclose(fit.predict(self.x), expected)
            with self.assertRaises(AttributeError):
                fit.__setattr__("intercept", 100.0)

    def test_strided_readonly_and_broadcast_inputs(self) -> None:
        for datafit_type, y in ((Gaussian, self.y), (Binomial, self.binary_y)):
            base_x, base_y = np.zeros((16, 8)), np.zeros(16)
            base_x[::2, ::2], base_y[::2] = self.x, y
            for solver in (Hybrid(), ProximalGradient()):
                expected = Problem(self.x, datafit_type(y), penalties()[0]).fit_with(
                    solver
                )
                for x_case, y_case in (
                    (np.asfortranarray(self.x), y.copy()),
                    (base_x[::2, ::2], base_y[::2]),
                    (self.x[::-1], y[::-1]),
                ):
                    x_case.flags.writeable = y_case.flags.writeable = False
                    actual = Problem(
                        x_case, datafit_type(y_case), penalties()[0]
                    ).fit_with(solver)
                    np.testing.assert_allclose(
                        actual.coefficients, expected.coefficients, atol=1e-8
                    )
                    self.assertFalse(x_case.flags.writeable)
                    self.assertFalse(y_case.flags.writeable)
                x_broadcast = np.broadcast_to(self.x[:1], self.x.shape)
                y_broadcast = np.broadcast_to(y[:1], y.shape)
                fit = Problem(
                    x_broadcast, datafit_type(y_broadcast), penalties()[0]
                ).fit_with(solver)
                self.assertTrue(np.isfinite(fit.predict(x_broadcast)).all())

    def test_empty_predictions_zero_features_and_single_observation(self) -> None:
        for solver in (Hybrid(), ProximalGradient()):
            for datafit in (Gaussian(self.y), Binomial(self.binary_y)):
                y = cast(FloatArray, datafit.y)
                expected = float(y.mean())
                if isinstance(datafit, Binomial):
                    expected = math.log(expected / (1 - expected))
                for penalty in (
                    ClusteredLasso(0.1, 0.2),
                    ChainFusedLasso(0.1, 0.2, weights=[]),
                    Slope([]),
                ):
                    fit = Problem(np.empty((8, 0)), datafit, penalty).fit_with(solver)
                    self.assertEqual(fit.coefficients.shape, (0,))
                    np.testing.assert_allclose(
                        fit.predict(np.empty((3, 0))), expected, atol=1e-5
                    )
                    empty = fit.predict(np.empty((0, 0)))
                    self.assertEqual(empty.shape, (0,))
                    self.assertEqual(empty.dtype, np.float64)
            for family in (Gaussian, Binomial):
                fit = Problem(
                    self.x[:1], family(np.array([0.4])), penalties()[0]
                ).fit_with(solver)
                self.assertEqual(fit.predict(np.empty((0, 4))).shape, (0,))

    def test_dense_contract_and_compatibility_errors(self) -> None:
        for solver in (Hybrid(), ProximalGradient()):
            for datafit_type, y in ((Gaussian, self.y), (Binomial, self.binary_y)):
                fit = Problem(self.x, datafit_type(y), penalties()[0]).fit_with(solver)
                for x in (
                    self.x.tolist(),
                    self.x.astype(np.float32),
                    self.x.view(np.ma.MaskedArray),
                ):
                    with self.assertRaises(TypeError):
                        Problem(x, datafit_type(y), penalties()[0]).fit_with(solver)
                    with self.assertRaises(TypeError):
                        fit.predict(x)
                for bad_y in (y.tolist(), y.astype(np.float32)):
                    with self.assertRaises(TypeError):
                        Problem(self.x, datafit_type(bad_y), penalties()[0]).fit_with(
                            solver
                        )
                for bad_y in (y[:, None], y[:-1], np.full(8, np.nan)):
                    with self.assertRaises(ValueError):
                        Problem(self.x, datafit_type(bad_y), penalties()[0]).fit_with(
                            solver
                        )
                with self.assertRaisesRegex(ValueError, "copy=False"):
                    Problem(self.x, datafit_type(y), penalties()[0]).fit_with(
                        solver, copy=False
                    )
                with self.assertRaisesRegex(ValueError, "copy=False"):
                    fit.predict(self.x, copy=False)
                with self.assertRaisesRegex(ValueError, "columns; expected 4"):
                    fit.predict(self.x[:, :3])
                for x in (self.x[0], np.full((8, 4), np.inf)):
                    with self.assertRaises(ValueError):
                        fit.predict(x)
        problem = Problem(self.x, Gaussian(self.y), penalties()[0])
        with self.assertRaisesRegex(TypeError, "solver must be"):
            problem.fit_with(cast(Any, object()))
        with self.assertRaisesRegex(TypeError, "datafit must be"):
            Problem(self.x, cast(Any, object()), penalties()[0]).fit_with(Hybrid())
        with self.assertRaisesRegex(TypeError, "penalty must be"):
            Problem(self.x, Gaussian(self.y), cast(Any, object())).fit_with(Hybrid())
        with self.assertRaisesRegex(ValueError, "normalization must be"):
            Problem(
                self.x,
                Gaussian(self.y),
                penalties()[0],
                normalization=cast(Normalization, "bad"),
            ).fit_with(Hybrid())

    def test_invalid_penalties_and_binomial_responses(self) -> None:
        invalid: tuple[Penalty, ...] = (
            ClusteredLasso(-1, 0),
            ClusteredLasso(0, np.nan),
            ChainFusedLasso(np.inf, 0),
            ChainFusedLasso(0, -1),
            ChainFusedLasso(0, 1, weights=[1]),
            ChainFusedLasso(0, 1, weights=[1, -1, 0]),
            Slope([1, 2, 0, 0]),
            Slope([1, np.nan, 0, 0]),
            Slope([1]),
        )
        for solver in (Hybrid(), ProximalGradient()):
            for penalty in invalid:
                with self.subTest(penalty=penalty), self.assertRaises(ValueError):
                    Problem(self.x, Gaussian(self.y), penalty).fit_with(solver)
            for bad in (-0.1, 1.1):
                y = self.binary_y.copy()
                y[0] = bad
                with self.assertRaisesRegex(ValueError, r"row 0.*\[0, 1\]"):
                    Problem(self.x, Binomial(y), penalties()[0]).fit_with(solver)
        for weights in ([[1, 2]], 1.0):
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                Slope(weights)
        with self.assertRaisesRegex(TypeError, "real numbers"):
            Slope(["bad"])

    def test_invalid_controls_and_solver_specific_stopping(self) -> None:
        problem = Problem(self.x, Gaussian(self.y), penalties()[0])
        for stopping in (
            StoppingCriterion.kkt_violation(),
            StoppingCriterion.duality_gap(),
        ):
            for solver_type in (Hybrid, ProximalGradient):
                with self.assertRaisesRegex(
                    ValueError, "unsupported fit combination.*mapping"
                ):
                    problem.fit_with(solver_type(stopping=stopping))
        with self.assertRaisesRegex(ValueError, "unsupported fit combination"):
            Lasso(0.1, stopping=StoppingCriterion.proximal_gradient_mapping()).fit(
                self.x, self.y
            )
        for value in (-1.0, 0.0, np.nan, np.inf):
            for solver_type in (Hybrid, ProximalGradient):
                with self.assertRaisesRegex(ValueError, "tolerance.*positive"):
                    problem.fit_with(
                        solver_type(
                            stopping=StoppingCriterion.proximal_gradient_mapping(value)
                        )
                    )
        for solver in (
            Hybrid(max_iterations=0),
            Hybrid(max_iterations=-1),
            Hybrid(cluster_sweeps=0),
            Hybrid(check_every=0),
            Hybrid(max_outer_iterations=0),
            Hybrid(max_inner_iterations=-1),
            Hybrid(max_outer_backtracks=-1),
            Hybrid(working_set_policy=WorkingSetPolicy.adaptive(0)),
            Hybrid(step_policy=HybridStep.backtracking(initial_curvature=0)),
            Hybrid(step_policy=HybridStep.backtracking(max_backtracks=-1)),
            ProximalGradient(max_iterations=0),
            ProximalGradient(initial_step=-1),
            ProximalGradient(max_backtracks=-1),
        ):
            with self.subTest(solver=solver), self.assertRaises(ValueError):
                problem.fit_with(solver)
        with self.assertRaisesRegex(ValueError, "relative must be zero"):
            problem.fit_with(
                Hybrid(
                    stopping=StoppingCriterion("proximal_gradient_mapping", 1e-6, 1e-6)
                )
            )

    def test_iteration_limits_and_line_search_errors(self) -> None:
        for datafit in (Gaussian(self.y), Binomial(self.binary_y)):
            problem = Problem(self.x, datafit, penalties()[0])
            for solver in (
                Hybrid(max_iterations=1),
                ProximalGradient(max_iterations=1),
            ):
                fit = problem.fit_with(solver)
                self.assertEqual(fit.termination, "iteration_limit")
                self.assertEqual(fit.iterations, 1)
                self.assertGreater(fit.stopping_value, fit.stopping_threshold)
                self.assertTrue(np.isfinite(fit.predict(self.x)).all())
            for solver in (
                Hybrid(
                    step_policy=HybridStep.backtracking(
                        initial_curvature=1e-20, max_backtracks=0
                    )
                ),
                ProximalGradient(initial_step=1e20, max_backtracks=0),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "line search failed.*iteration"
                ):
                    problem.fit_with(solver)

    def test_nonfinite_arithmetic_preserves_error_context(self) -> None:
        for solver in (Hybrid(), ProximalGradient()):
            with self.assertRaisesRegex(
                FloatingPointError, "nonfinite arithmetic.*iteration 0"
            ):
                Problem(
                    np.array([[1e308], [-1e308]]),
                    Gaussian(np.array([1.0, -1.0])),
                    ClusteredLasso(0.1, 0),
                    normalization="none",
                ).fit_with(solver)
            fit = Problem(
                np.array([[0.0], [1.0]]),
                Gaussian(np.array([0.0, 10.0])),
                ClusteredLasso(0, 0),
                normalization="none",
            ).fit_with(solver)
            with self.assertRaisesRegex(FloatingPointError, "predicting.*iteration"):
                fit.predict(np.array([[1e308]]))

    def _during_native(self, action: Callable[[], T], mutate: Callable[[], None]) -> T:
        ready, start, changed = threading.Event(), threading.Event(), threading.Event()

        def worker() -> None:
            ready.set()
            start.wait()
            # Scheduling proves detachment even if mutation outlasts prediction.
            changed.set()
            mutate()

        thread = threading.Thread(target=worker)
        thread.start()
        ready.wait()
        interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(10.0)
            start.set()
            result = action()
            ran_during_native = changed.is_set()
        finally:
            sys.setswitchinterval(interval)
            start.set()
            thread.join()
        self.assertTrue(
            ran_during_native, "a Python thread could not run during native work"
        )
        return result

    def test_native_fits_release_gil_after_snapshot_for_both_families(self) -> None:
        rng = np.random.default_rng(42)
        original = rng.normal(size=(10_000, 40))
        for family in ("gaussian", "binomial"):
            for solver_type in (Hybrid, ProximalGradient):
                x = original.copy()
                y = x @ np.linspace(-2, 2, 40)
                if family == "binomial":
                    y = 1 / (1 + np.exp(-y))
                native = solver_type(
                    max_iterations=100,
                    stopping=StoppingCriterion.proximal_gradient_mapping(1e-12),
                )._native()
                spec = ClusteredLasso(0.01, 0.001)._spec()
                expected = native.fit(x, y, family, True, "auto", spec)

                actual = self._during_native(
                    partial(native.fit, x, y, family, True, "auto", spec),
                    partial(zero_arrays, x, y),
                )
                np.testing.assert_array_equal(
                    actual.coefficients, expected.coefficients
                )
                self.assertEqual(actual.intercept, expected.intercept)

    def test_native_predictions_release_gil_after_snapshot(self) -> None:
        rng = np.random.default_rng(93)
        original = rng.normal(size=(100_000, 40))
        for solver in (Hybrid(), ProximalGradient()):
            fit = Problem(
                original[:100],
                Gaussian(original[:100] @ np.linspace(1, 2, 40)),
                ClusteredLasso(0, 0),
            ).fit_with(solver)
            x = original.copy()
            expected = fit._native.predict(x)
            actual = self._during_native(
                partial(fit._native.predict, x), partial(zero_arrays, x)
            )
            np.testing.assert_array_equal(actual, expected)

    def test_same_problem_and_solver_support_concurrent_fits(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        for solver in (Hybrid(), ProximalGradient()):
            for datafit in (Gaussian(self.y), Binomial(self.binary_y)):
                problem = Problem(self.x, datafit, penalties()[2])
                expected = problem.fit_with(solver)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(fit_problem, problem, solver) for _ in range(4)
                    ]
                    for future in futures:
                        fit = future.result()
                        np.testing.assert_array_equal(
                            fit.coefficients, expected.coefficients
                        )
                        self.assertEqual(fit.objective, expected.objective)

    def test_public_diagnostics_are_ordinary_immutable_records(self) -> None:
        fit = Problem(self.x, Gaussian(self.y), penalties()[0]).fit_with(Hybrid())
        self.assertIn("full_checks", asdict(fit.working_set_diagnostics))
        self.assertIn("certificate_passes", asdict(fit.screening_diagnostics))


if __name__ == "__main__":
    unittest.main()
