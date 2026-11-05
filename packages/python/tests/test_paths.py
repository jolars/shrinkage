"""Exercise warm paths through the installed Python distribution."""

from __future__ import annotations

import gc
import json
import math
import sys
import threading
import unittest
import weakref
from collections.abc import Callable
from dataclasses import replace
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
    PathFit,
    PathOptions,
    Problem,
    ProximalFit,
    ProximalGradient,
    ScreeningDiagnostics,
    Slope,
    StoppingCriterion,
    WorkingSetDiagnostics,
    WorkingSetPolicy,
)

Penalty = ClusteredLasso | ChainFusedLasso | Slope
T = TypeVar("T")


def zero_arrays(*arrays: npt.NDArray[np.float64]) -> None:
    for array in arrays:
        array.fill(0)


def penalty_from_record(record: dict[str, Any]) -> Penalty:
    options = record.copy()
    kind = options.pop("kind")
    if kind == "clustered_lasso":
        return ClusteredLasso(**options)
    if kind == "chain_fused_lasso":
        return ChainFusedLasso(**options)
    assert kind == "slope"
    return Slope(**options)


def solver_from_record(case: dict[str, Any]) -> Hybrid | ProximalGradient:
    options = case["solver_options"].copy()
    options["stopping"] = StoppingCriterion(**options["stopping"])
    if case["solver"] == "hybrid":
        options["working_set_policy"] = WorkingSetPolicy(
            **options["working_set_policy"]
        )
        options["step_policy"] = HybridStep(**options["step_policy"])
        return Hybrid(**options)
    return ProximalGradient(**options)


class PathTests(unittest.TestCase):
    def test_all_path_metadata_and_point_diagnostics_match_rust(self) -> None:
        fixture = json.loads(
            Path(__file__).with_name("fixtures").joinpath("path_rust.json").read_text()
        )
        x = np.asarray(fixture["x"], dtype=np.float64)
        predict_x = np.asarray(fixture["predict_x"], dtype=np.float64)
        combinations, modes, failures = set(), set(), set()
        warm_improvements = set()
        screened_combinations, reactivated_combinations = set(), set()
        for case in fixture["cases"]:
            with self.subTest(
                family=case["family"],
                solver=case["solver"],
                mode=case["mode"],
                penalty=case["penalty"],
                options=case["solver_options"],
            ):
                y = np.asarray(fixture[f"{case['family']}_y"], dtype=np.float64)
                if case["mode"] == "boundary":
                    y.fill(1)
                family = Gaussian if case["family"] == "gaussian" else Binomial
                penalty = penalty_from_record(case["penalty"])
                problem = Problem(
                    x,
                    family(y, fit_intercept=case["fit_intercept"]),
                    penalty,
                    normalization=case["normalization"],
                )
                solver = solver_from_record(case)
                action: Callable[[], PathFit[HybridFit] | PathFit[ProximalFit]]
                if case["penalties"] is None:
                    action = partial(
                        problem.fit_regularization_path_with,
                        solver,
                        PathOptions(**case["path_options"]),
                    )
                else:
                    action = partial(
                        problem.fit_path_with,
                        solver,
                        [penalty_from_record(record) for record in case["penalties"]],
                    )
                expected = case["expected"]
                if "error" in expected:
                    error = expected["error"]
                    exception = {
                        "ValueError": ValueError,
                        "RuntimeError": RuntimeError,
                        "FloatingPointError": FloatingPointError,
                    }[error["kind"]]
                    with self.assertRaises(exception) as context:
                        action()
                    self.assertEqual(str(context.exception), error["message"])
                    failures.add(case["mode"])
                    continue
                path = action()
                combination = (case["family"], case["solver"], type(penalty).__name__)
                combinations.add(combination)
                modes.add(case["mode"])
                self.assertEqual(path.family, case["family"])
                self.assertEqual(path.solver, case["solver"])
                self.assertEqual(path.n_features, x.shape[1])
                self.assertEqual(
                    path.penalties,
                    tuple(
                        penalty_from_record(record) for record in expected["penalties"]
                    ),
                )
                self.assertEqual(path.termination, expected["termination"])
                self.assertEqual(len(path), len(expected["fits"]))
                for name in (
                    "lambdas",
                    "maximum_penalty",
                    "lambda_start",
                    "null_deviance",
                    "deviances",
                ):
                    value = expected[name]
                    if isinstance(value, str) and value == "unbounded":
                        self.assertEqual(path.maximum_penalty, math.inf)
                    elif value is None:
                        self.assertIsNone(getattr(path, name))
                    else:
                        np.testing.assert_allclose(
                            getattr(path, name), value, rtol=1e-12, atol=1e-12
                        )
                self.assertEqual(path.coefficients.shape, (len(path), x.shape[1]))
                self.assertEqual(path.intercepts.shape, (len(path),))
                self.assertEqual(path.predict(predict_x).shape, (2, len(path)))
                for index, (fit, point) in enumerate(
                    zip(path.fits, expected["fits"], strict=True)
                ):
                    self._assert_point(fit, point, predict_x)
                    if fit.screening_diagnostics.screened_features > 0:
                        screened_combinations.add(combination)
                    if (
                        isinstance(fit, HybridFit)
                        and fit.working_set_diagnostics.reactivations > 0
                    ):
                        reactivated_combinations.add(combination)
                    np.testing.assert_array_equal(
                        path.coefficients[index], fit.coefficients
                    )
                    self.assertEqual(path.intercepts[index], fit.intercept)
                    np.testing.assert_allclose(
                        path.predict(predict_x)[:, index],
                        point["predictions"],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                if case["mode"] == "limited":
                    self.assertEqual(path.fits[1].termination, "iteration_limit")
                    self.assertEqual(path.fits[2].termination, "iteration_limit")
                    self.assertLess(path.fits[2].objective, path.fits[1].objective)
                    warm_improvements.add(combination)
                if case["mode"] == "early_change":
                    self.assertEqual(path.termination, "deviance_change")
                    self.assertEqual(len(path), 2)
                if case["mode"] == "early_ratio":
                    self.assertEqual(path.termination, "deviance_ratio")
                    self.assertEqual(len(path), 1)
                if case["mode"] == "limited_auto":
                    self.assertEqual(path.termination, "completed")
                    self.assertEqual(len(path), 6)
                    self.assertTrue(
                        all(fit.termination == "iteration_limit" for fit in path.fits)
                    )
        self.assertEqual(len(combinations), 12)
        self.assertEqual(warm_improvements, combinations)
        self.assertEqual(screened_combinations, combinations)
        self.assertEqual(
            reactivated_combinations,
            {item for item in combinations if item[1] == "hybrid"},
        )
        self.assertTrue(
            {"explicit", "automatic", "override", "early_change", "early_ratio"}
            <= modes
        )
        self.assertEqual(
            failures, {"invalid_point", "line_search", "unbounded", "boundary"}
        )

    def _assert_point(self, fit: ProximalFit, expected: dict[str, Any], x: Any) -> None:
        for name, value in expected.items():
            if name in ("centers", "scales"):
                actual = getattr(fit.preprocessing, name)
            elif name == "predictions":
                actual = fit.predict(x)
            elif name == "screening_diagnostics":
                self.assertEqual(
                    fit.screening_diagnostics, ScreeningDiagnostics(**value)
                )
                continue
            elif name == "working_set_diagnostics":
                value = value.copy()
                value["policy"] = WorkingSetPolicy(**value["policy"])
                assert isinstance(fit, HybridFit)
                self.assertEqual(
                    fit.working_set_diagnostics, WorkingSetDiagnostics(**value)
                )
                continue
            elif name == "step_policy":
                assert isinstance(fit, HybridFit)
                self.assertEqual(fit.step_policy, HybridStep(**value))
                continue
            else:
                actual = getattr(fit, name)
            if value is None or isinstance(value, str | int):
                self.assertEqual(actual, value, name)
            else:
                np.testing.assert_allclose(
                    actual, value, rtol=1e-10, atol=1e-12, err_msg=name
                )

    def test_explicit_paths_preserve_points_shapes_and_predictions(self) -> None:
        x = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        for datafit in (
            Gaussian(np.array([-1.0, -7.0, 9.0, 3.0])),
            Binomial(np.array([0.1, 0.4, 0.6, 0.9])),
        ):
            point_sequences: tuple[list[Penalty], ...] = (
                [ClusteredLasso(1.0, 2.0), ClusteredLasso(0.1, 0.01)],
                [
                    ChainFusedLasso(1.0, 2.0, weights=[0.0]),
                    ChainFusedLasso(0.1, 0.01, weights=[2.0]),
                ],
                [Slope([1.0, 0.5]), Slope([0.1, 0.0])],
            )
            for points in point_sequences:
                points += [points[0], points[0]]
                for solver in (
                    Hybrid(stopping=StoppingCriterion.proximal_gradient_mapping(1e-9)),
                    ProximalGradient(
                        stopping=StoppingCriterion.proximal_gradient_mapping(1e-9)
                    ),
                ):
                    with self.subTest(datafit=datafit, points=points, solver=solver):
                        problem = Problem(x, datafit, points[0], normalization="none")
                        path = problem.fit_path_with(solver, points)
                        self.assertIsInstance(path, PathFit)
                        self.assertEqual(path.penalties, tuple(points))
                        self.assertEqual(len(path), 4)
                        self.assertEqual(path.coefficients.shape, (4, 2))
                        self.assertEqual(path.intercepts.shape, (4,))
                        self.assertEqual(path.predict(x).shape, (4, 4))
                        self.assertEqual(path.predict(x[:1]).shape, (1, 4))
                        self.assertEqual(path.predict(x[:0]).shape, (0, 4))
                        self.assertEqual(path.termination, "completed")
                        self.assertIsNone(path.lambdas)
                        self.assertIsNone(path.maximum_penalty)
                        self.assertIsNone(path.lambda_start)
                        self.assertIsNone(path.null_deviance)
                        self.assertIsNone(path.deviances)
                        for index, (fit, penalty) in enumerate(
                            zip(path.fits, points, strict=True)
                        ):
                            cold = Problem(
                                x, datafit, penalty, normalization="none"
                            ).fit_with(solver)
                            self.assertEqual(fit.termination, "converged")
                            np.testing.assert_allclose(
                                path.predict(x)[:, index], fit.predict(x)
                            )
                            np.testing.assert_allclose(
                                fit.predict(x), cold.predict(x), atol=1e-6
                            )
                            if isinstance(fit, HybridFit):
                                self.assertEqual(
                                    fit.working_set_diagnostics.policy.kind,
                                    "ever_active",
                                )

    def test_automatic_paths_keep_complete_scaled_penalties_and_metadata(self) -> None:
        x = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        options = PathOptions(nlambda=4, lambda_min_ratio=0.05, early_stopping=False)
        for datafit in (
            Gaussian(np.array([-1.0, -7.0, 9.0, 3.0])),
            Binomial(np.array([0.1, 0.4, 0.6, 0.9])),
        ):
            for penalty in (
                ClusteredLasso(2.0, 1.0),
                ChainFusedLasso(2.0, 1.0, weights=[0.0]),
                Slope([2.0, 0.5]),
            ):
                for solver in (Hybrid(), ProximalGradient()):
                    path = Problem(x, datafit, penalty).fit_regularization_path_with(
                        solver, options
                    )
                    self.assertEqual(len(path), 4)
                    assert path.lambdas is not None
                    self.assertEqual(path.lambdas.shape, (4,))
                    self.assertEqual(path.lambda_start, path.maximum_penalty)
                    self.assertEqual(path.lambda_start, path.lambdas[0])
                    self.assertAlmostEqual(path.lambdas[-1] / path.lambdas[0], 0.05)
                    self.assertIsNotNone(path.null_deviance)
                    assert path.deviances is not None
                    self.assertEqual(path.deviances.shape, (4,))
                    np.testing.assert_allclose(path.coefficients[0], 0, atol=1e-12)
                    for strength, actual in zip(
                        path.lambdas, path.penalties, strict=True
                    ):
                        if isinstance(penalty, Slope):
                            assert isinstance(actual, Slope)
                            np.testing.assert_allclose(
                                actual.weights, strength * np.asarray(penalty.weights)
                            )
                        else:
                            assert not isinstance(actual, Slope)
                            self.assertAlmostEqual(actual.l1, strength * penalty.l1)
                            self.assertAlmostEqual(
                                actual.fusion, strength * penalty.fusion
                            )
                            if isinstance(penalty, ChainFusedLasso):
                                assert isinstance(actual, ChainFusedLasso)
                                self.assertEqual(actual.weights, penalty.weights)

    def test_empty_paths_batches_and_zero_features_keep_both_axes(self) -> None:
        for solver in (Hybrid(), ProximalGradient()):
            for nfeatures in (0, 2):
                x = np.zeros((3, nfeatures))
                for family in (Gaussian, Binomial):
                    for penalty in (
                        ClusteredLasso(1, 1),
                        ChainFusedLasso(1, 1),
                        Slope(np.ones(nfeatures)),
                    ):
                        problem = Problem(x, family(np.array([0.2, 0.5, 0.8])), penalty)
                        empty = problem.fit_path_with(solver, [])
                        self.assertEqual(empty.fits, ())
                        self.assertEqual(empty.penalties, ())
                        self.assertEqual(empty.coefficients.shape, (0, nfeatures))
                        self.assertEqual(empty.intercepts.shape, (0,))
                        self.assertEqual(empty.predict(x).shape, (3, 0))
                        self.assertEqual(empty.predict(x[:0]).shape, (0, 0))
                        self.assertEqual(empty.predict(x).dtype, np.float64)
                        with self.assertRaisesRegex(ValueError, "columns; expected"):
                            empty.predict(np.zeros((0, nfeatures + 1)))
                        path = problem.fit_regularization_path_with(
                            solver, PathOptions(nlambda=2, early_stopping=False)
                        )
                        self.assertEqual(path.coefficients.shape, (2, nfeatures))
                        self.assertEqual(path.predict(x[:0]).shape, (0, 2))
                        np.testing.assert_array_equal(path.lambdas, [0.0, 0.0])

    def test_default_endpoint_ratio_depends_on_training_dimensions(self) -> None:
        for nfeatures, ratio in ((1, 0.0001), (4, 0.01)):
            x = np.arange(3 * nfeatures, dtype=np.float64).reshape(3, nfeatures)
            for solver in (Hybrid(), ProximalGradient()):
                path = Problem(
                    x, Gaussian(np.array([0.0, 1.0, 2.0])), ClusteredLasso(1, 0)
                ).fit_regularization_path_with(
                    solver, PathOptions(nlambda=2, early_stopping=False)
                )
                assert path.lambdas is not None
                self.assertAlmostEqual(path.lambdas[-1] / path.lambdas[0], ratio)

    def test_invalid_options_inputs_controls_and_mixed_penalties(self) -> None:
        x, y = np.array([[-1.0], [0.0], [1.0]]), np.array([0.1, 0.3, 0.8])
        penalty = ClusteredLasso(1, 0)
        for solver in (Hybrid(), ProximalGradient()):
            problem = Problem(x, Gaussian(y), penalty)
            for changes in (
                {"nlambda": 0},
                {"nlambda": -1},
                {"min_points": 0},
                {"lambda_min_ratio": 0.0},
                {"lambda_min_ratio": 1.1},
                {"lambda_min_ratio": np.nan},
                {"lambda_start": -1.0},
                {"lambda_start": np.inf},
                {"deviance_change_tolerance": -1.0},
                {"deviance_change_tolerance": np.nan},
                {"deviance_ratio_tolerance": 0.0},
                {"deviance_ratio_tolerance": 1.1},
                {"deviance_ratio_tolerance": np.nan},
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    problem.fit_regularization_path_with(
                        solver, replace(PathOptions(), **cast(Any, changes))
                    )
            with self.assertRaisesRegex(TypeError, "options must be"):
                problem.fit_regularization_path_with(solver, cast(Any, object()))
            for points in ([penalty, ChainFusedLasso(1, 0)], [object()]):
                with self.assertRaisesRegex(TypeError, "same type"):
                    problem.fit_path_with(solver, cast(Any, points))
            for automatic in (False, True):

                def fit_inputs(
                    x_case: object,
                    y_case: object,
                    copy: bool = True,
                    automatic: bool = automatic,
                    solver: Hybrid | ProximalGradient = solver,
                ) -> PathFit[HybridFit] | PathFit[ProximalFit]:
                    problem = Problem(x_case, Gaussian(y_case), penalty)
                    if automatic:
                        return problem.fit_regularization_path_with(
                            solver, PathOptions(nlambda=2), copy=copy
                        )
                    return problem.fit_path_with(solver, [penalty], copy=copy)

                with self.assertRaisesRegex(ValueError, "copy=False"):
                    fit_inputs(x, y, False)
                for x_case, y_case in (
                    (x.tolist(), y),
                    (x.astype(np.float32), y),
                    (x, y.astype(np.float32)),
                    (x, y.tolist()),
                ):
                    with self.assertRaises(TypeError):
                        fit_inputs(x_case, y_case)
                for x_case, y_case in (
                    (x[0], y),
                    (x, y[:-1]),
                    (x, y[:, None]),
                    (x, y * np.nan),
                ):
                    with self.assertRaises(ValueError):
                        fit_inputs(x_case, y_case)
            path = problem.fit_path_with(solver, [penalty])
            with self.assertRaisesRegex(ValueError, "copy=False"):
                path.predict(x, copy=False)
            with self.assertRaises(TypeError):
                path.predict(x.astype(np.float32))
            with self.assertRaises(ValueError):
                path.predict(x * np.nan)
        for solver in (
            Hybrid(stopping=StoppingCriterion.kkt_violation()),
            ProximalGradient(stopping=StoppingCriterion.duality_gap()),
        ):
            problem = Problem(x, Gaussian(y), penalty)
            for automatic in (False, True):
                with self.assertRaisesRegex(ValueError, "unsupported fit combination"):
                    if automatic:
                        problem.fit_regularization_path_with(solver)
                    else:
                        problem.fit_path_with(solver, [penalty])

    def test_nonfinite_path_and_prediction_arithmetic_keep_rust_context(self) -> None:
        for solver in (Hybrid(), ProximalGradient()):
            problem = Problem(
                np.array([[1e308], [-1e308]]),
                Gaussian(np.array([1.0, -1.0])),
                ClusteredLasso(0.1, 0),
                normalization="none",
            )
            for automatic in (False, True):
                with self.assertRaisesRegex(
                    FloatingPointError, "nonfinite arithmetic.*iteration 0"
                ):
                    if automatic:
                        problem.fit_regularization_path_with(
                            solver, PathOptions(nlambda=2)
                        )
                    else:
                        problem.fit_path_with(solver, [ClusteredLasso(0.1, 0)])
            path = Problem(
                np.array([[0.0], [1.0]]),
                Gaussian(np.array([0.0, 10.0])),
                ClusteredLasso(0, 0),
                normalization="none",
            ).fit_path_with(solver, [ClusteredLasso(0, 0)])
            with self.assertRaisesRegex(FloatingPointError, "predicting.*iteration"):
                path.predict(np.array([[1e308]]))

    def test_paths_and_individual_points_outlive_their_inputs(self) -> None:
        for solver in (Hybrid(), ProximalGradient()):
            x = np.array([[-2.0, 1.0], [0.0, -1.0], [2.0, 1.0]])
            y = np.array([-2.0, 0.1, 2.0])
            references = (weakref.ref(x), weakref.ref(y))
            problem = Problem(x, Gaussian(y), Slope([1.0, 0.5]))
            path = problem.fit_regularization_path_with(
                solver, PathOptions(nlambda=3, early_stopping=False)
            )
            prediction_x = x.copy()
            expected = path.predict(prediction_x)
            point = path.fits[-1]
            for array in (
                path.coefficients,
                path.intercepts,
                path.lambdas,
                path.deviances,
                path.predict(x),
                point.coefficients,
            ):
                assert array is not None
                array.fill(-999)
            x.fill(999)
            y.fill(999)
            del x, y, problem
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            np.testing.assert_array_equal(path.predict(prediction_x), expected)
            with self.assertRaises(AttributeError):
                path.__setattr__("termination", "completed")
            del path
            np.testing.assert_array_equal(point.predict(prediction_x), expected[:, -1])

    def _during_native(self, action: Callable[[], T], mutate: Callable[[], None]) -> T:
        start, changed = threading.Event(), threading.Event()

        def worker() -> None:
            start.wait()
            changed.set()
            mutate()

        thread = threading.Thread(target=worker)
        thread.start()
        interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(10.0)
            start.set()
            result = action()
            ran = changed.is_set()
        finally:
            sys.setswitchinterval(interval)
            start.set()
            thread.join()
        self.assertTrue(ran, "a Python thread could not run during native work")
        return result

    def test_paths_and_bulk_predictions_detach_after_copying_inputs(self) -> None:
        rng = np.random.default_rng(928)
        original = rng.normal(size=(10_000, 40))
        spec = ClusteredLasso(0.01, 0.001)._spec()
        for family in ("gaussian", "binomial"):
            for solver in (
                Hybrid(max_iterations=50),
                ProximalGradient(max_iterations=50),
            ):
                native = solver._native()
                x = original.copy()
                y = x @ np.linspace(-2, 2, 40)
                if family == "binomial":
                    y = 1 / (1 + np.exp(-y))
                action = partial(
                    native.path, x, y, family, True, "auto", spec, penalties=[spec] * 3
                )
                expected = action()

                actual = self._during_native(action, partial(zero_arrays, x, y))
                np.testing.assert_array_equal(
                    actual.coefficients, expected.coefficients
                )
                np.testing.assert_array_equal(actual.intercepts, expected.intercepts)
                prediction_x = np.tile(original, (5, 1))
                expected_predictions = actual.predict(prediction_x)
                actual_predictions = self._during_native(
                    partial(actual.predict, prediction_x), partial(prediction_x.fill, 0)
                )
                self.assertEqual(actual_predictions, expected_predictions)


if __name__ == "__main__":
    unittest.main()
