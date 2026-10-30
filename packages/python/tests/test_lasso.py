"""Exercise Gaussian lasso through the installed public package."""

from __future__ import annotations

import gc
import json
import sys
import threading
import unittest
import weakref
from pathlib import Path
from typing import Any, cast

import numpy as np

from shrinkage import Lasso, Normalization, StoppingCriterion, _shrinkage


class LassoTests(unittest.TestCase):
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

    def test_analytical_fit_and_prediction(self) -> None:
        x = np.array([[-1.0], [0.0], [1.0]])
        y = 2.0 * x[:, 0] + 3.0
        fit = Lasso(
            0.2,
            normalization="none",
            stopping=StoppingCriterion.kkt_violation(1e-12),
        ).fit(x, y)
        np.testing.assert_allclose(fit.coefficients, [1.7], atol=1e-12)
        self.assertAlmostEqual(fit.intercept, 3.0)
        self.assertEqual(fit.termination, "converged")
        self.assertLessEqual(fit.stopping_value, fit.stopping_threshold)
        np.testing.assert_allclose(fit.predict(x), x @ fit.coefficients + fit.intercept)

    def test_all_diagnostics_match_public_rust_fixtures(self) -> None:
        fixture = json.loads(
            Path(__file__).with_name("fixtures").joinpath("lasso_rust.json").read_text()
        )
        x = np.asarray(fixture["x"], dtype=np.float64)
        y = np.asarray(fixture["y"], dtype=np.float64)
        predict_x = np.asarray(fixture["predict_x"], dtype=np.float64)
        self.assertEqual(len(fixture["cases"]), 21)
        for case in fixture["cases"]:
            with self.subTest(case=case):
                stopping = case["stopping"]
                fit = Lasso(
                    case["lambda"],
                    fit_intercept=case["fit_intercept"],
                    normalization=case["normalization"],
                    stopping=None
                    if stopping is None
                    else StoppingCriterion(**stopping),
                    max_iterations=case["max_iterations"],
                ).fit(x, y)
                expected = case["expected"]
                for field in (
                    "coefficients",
                    "intercept",
                    "objective",
                    "kkt_violation",
                    "dual_certificate",
                    "dual_objective",
                    "duality_gap",
                    "reference_loss",
                    "stopping_value",
                    "stopping_threshold",
                ):
                    np.testing.assert_allclose(
                        getattr(fit, field),
                        expected[field],
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=field,
                    )
                self.assertEqual(fit.termination, expected["termination"])
                self.assertEqual(fit.iterations, expected["iterations"])
                self.assertEqual(
                    fit.stopping, StoppingCriterion(**expected["stopping"])
                )
                for field in ("centers", "scales"):
                    actual = getattr(fit.preprocessing, field)
                    if expected[field] is None:
                        self.assertIsNone(actual)
                    else:
                        np.testing.assert_allclose(actual, expected[field], atol=1e-12)
                np.testing.assert_allclose(
                    fit.predict(predict_x), expected["predictions"], atol=1e-12
                )

    def test_optimization_scale_objective_and_optimality(self) -> None:
        for normalization in ("auto", "none", "standardize", "min_max", "l2"):
            for fit_intercept in (True, False):
                with self.subTest(normalization=normalization, intercept=fit_intercept):
                    fit = Lasso(
                        0.12,
                        normalization=normalization,
                        fit_intercept=fit_intercept,
                        stopping=StoppingCriterion.kkt_violation(1e-10),
                    ).fit(self.x, self.y)
                    preprocessing = fit.preprocessing
                    centers = preprocessing.centers
                    scales = preprocessing.scales
                    x_tilde = self.x - (0.0 if centers is None else centers)
                    scales = np.ones(4) if scales is None else scales
                    x_tilde /= scales
                    beta = fit.coefficients * scales
                    residual = self.y - fit.predict(self.x)
                    correlations = x_tilde.T @ residual / len(self.y)
                    violations = np.where(
                        beta != 0.0,
                        abs(correlations - 0.12 * np.sign(beta)),
                        np.maximum(abs(correlations) - 0.12, 0.0),
                    )
                    kkt = float(np.max(violations))
                    if fit_intercept:
                        kkt = max(kkt, abs(float(residual.mean())))
                    self.assertAlmostEqual(fit.kkt_violation, kkt, delta=1e-12)
                    self.assertLessEqual(kkt, 1.1e-10)
                    objective = residual @ residual / (2 * len(self.y))
                    objective += 0.12 * abs(beta).sum()
                    self.assertAlmostEqual(fit.objective, objective, delta=1e-12)
                    u = fit.dual_certificate
                    self.assertLessEqual(
                        float(np.max(abs(x_tilde.T @ u))), 0.12 + 1e-12
                    )
                    if fit_intercept:
                        self.assertAlmostEqual(float(u.sum()), 0.0, delta=1e-12)
                    dual = self.y @ u - len(self.y) * (u @ u) / 2
                    self.assertAlmostEqual(fit.dual_objective, dual, delta=1e-12)
                    self.assertAlmostEqual(
                        fit.duality_gap, objective - dual, delta=1e-12
                    )

    def test_preprocessing_is_fitted_once_and_predictions_use_original_scale(
        self,
    ) -> None:
        fit = Lasso(0.12).fit(self.x, self.y)
        preprocessing = fit.preprocessing
        self.assertEqual(preprocessing.centering, "mean")
        self.assertEqual(preprocessing.scaling, "sd")
        assert preprocessing.centers is not None and preprocessing.scales is not None
        np.testing.assert_allclose(preprocessing.centers, self.x.mean(axis=0))
        expected_scales = self.x.std(axis=0)
        expected_scales[expected_scales == 0] = 1
        np.testing.assert_allclose(preprocessing.scales, expected_scales)
        new_x = self.x[:3] + 100.0
        expected = new_x @ fit.coefficients + fit.intercept
        np.testing.assert_allclose(fit.predict(new_x), expected)
        for row, prediction in zip(new_x, expected, strict=True):
            np.testing.assert_allclose(fit.predict(row[None, :]), [prediction])
        no_intercept = Lasso(0.12, fit_intercept=False).fit(self.x, self.y)
        self.assertIsNone(no_intercept.preprocessing.centers)
        self.assertEqual(no_intercept.preprocessing.centering, "none")
        raw = Lasso(0.12, normalization="none").fit(self.x, self.y)
        self.assertIsNone(raw.preprocessing.centers)
        self.assertIsNone(raw.preprocessing.scales)

    def test_iteration_limit_returns_finite_parameters_without_convergence(
        self,
    ) -> None:
        fit = Lasso(
            0.12,
            normalization="none",
            max_iterations=1,
            stopping=StoppingCriterion.kkt_violation(1e-15),
        ).fit(self.x, self.y)
        self.assertEqual(fit.termination, "iteration_limit")
        self.assertEqual(fit.iterations, 1)
        self.assertGreater(fit.stopping_value, fit.stopping_threshold)
        self.assertTrue(np.isfinite(fit.coefficients).all())
        self.assertTrue(np.isfinite(fit.predict(self.x)).all())
        self.assertTrue(np.isfinite(fit.objective))

    def test_resolved_defaults_and_explicit_gap_tolerances(self) -> None:
        for strength in (0.0, 0.12):
            fit = Lasso(strength).fit(self.x, self.y)
            expected = (
                StoppingCriterion.kkt_violation()
                if strength == 0.0
                else StoppingCriterion.duality_gap()
            )
            self.assertEqual(fit.stopping, expected)
        stopping = StoppingCriterion.duality_gap(1e-8, absolute=1e-11)
        fit = Lasso(0.12, stopping=stopping).fit(self.x, self.y)
        self.assertEqual(fit.stopping, stopping)
        self.assertEqual(fit.stopping_value, fit.duality_gap)
        self.assertAlmostEqual(
            fit.stopping_threshold, 1e-11 + 1e-8 * fit.reference_loss, delta=1e-20
        )
        null = Lasso(100.0).fit(self.x, self.y)
        self.assertEqual(null.iterations, 0)

    def test_layouts_and_readonly_inputs_match_contiguous_fit(self) -> None:
        base = np.zeros((16, 8))
        base[::2, ::2] = self.x
        response = np.zeros(16)
        response[::2] = self.y
        expected = Lasso(0.12).fit(self.x, self.y)
        for x, y in (
            (self.x, self.y),
            (np.asfortranarray(self.x), self.y),
            (self.x.T.copy().T, self.y),
            (base[::2, ::2], response[::2]),
            (self.x[::-1, ::-1], self.y[::-1]),
        ):
            with self.subTest(strides=x.strides):
                x.flags.writeable = y.flags.writeable = False
                fit = Lasso(0.12).fit(x, y)
                order = slice(None, None, -1) if x.strides[1] < 0 else slice(None)
                np.testing.assert_allclose(
                    fit.coefficients, expected.coefficients[order], atol=1e-5
                )
                np.testing.assert_allclose(
                    fit.predict(x), x @ fit.coefficients + fit.intercept
                )
                self.assertFalse(x.flags.writeable)
                self.assertFalse(y.flags.writeable)
        x = np.broadcast_to(self.x[:1], self.x.shape)
        y = np.broadcast_to(self.y[:1], self.y.shape)
        fit = Lasso(0.12).fit(x, y)
        np.testing.assert_allclose(fit.predict(x), y)

    def test_empty_prediction_batches_zero_features_and_single_observation(
        self,
    ) -> None:
        for intercept in (True, False):
            fit = Lasso(0.1, fit_intercept=intercept).fit(np.empty((8, 0)), self.y)
            self.assertEqual(fit.coefficients.shape, (0,))
            self.assertEqual(fit.coefficients.dtype, np.float64)
            expected = self.y.mean() if intercept else 0.0
            np.testing.assert_allclose(fit.predict(np.empty((3, 0))), [expected] * 3)
            self.assertEqual(fit.predict(np.empty((0, 0))).shape, (0,))
        fit = Lasso(0.1).fit(self.x[:1], self.y[:1])
        self.assertEqual(fit.predict(np.empty((0, 4))).shape, (0,))
        self.assertEqual(fit.predict(np.empty((0, 4))).dtype, np.float64)
        np.testing.assert_allclose(fit.predict(self.x[:1]), self.y[:1])

    def test_fit_outlives_sources_and_returned_arrays_are_independent(self) -> None:
        x, y = self.x.copy(), self.y.copy()
        references = [weakref.ref(x), weakref.ref(y)]
        fit = Lasso(0.12).fit(x, y)
        predictions = fit.predict(self.x)
        x[:] = y[:] = -100.0
        del x, y
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        fit.coefficients[:] = 1000.0
        fit.dual_certificate[:] = 1000.0
        preprocessing = fit.preprocessing
        assert preprocessing.centers is not None and preprocessing.scales is not None
        preprocessing.centers[:] = 1000.0
        preprocessing.scales[:] = 1000.0
        predictions[:] = -1000.0
        expected = self.x @ fit.coefficients + fit.intercept
        np.testing.assert_allclose(fit.predict(self.x), expected)
        fitted_centers = fit.preprocessing.centers
        assert fitted_centers is not None
        np.testing.assert_allclose(fitted_centers, self.x.mean(axis=0))
        with self.assertRaises(AttributeError):
            fit.__setattr__("intercept", 0.0)

    def test_invalid_options_raise_actionable_value_errors(self) -> None:
        for strength in (-1.0, np.inf, np.nan):
            with (
                self.subTest(strength=strength),
                self.assertRaisesRegex(ValueError, "finite.*nonnegative"),
            ):
                Lasso(strength).fit(self.x, self.y)
        for budget in (-1, 0):
            with self.assertRaisesRegex(ValueError, "max_iterations.*positive"):
                Lasso(0.1, max_iterations=budget).fit(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "normalization must be one of"):
            Lasso(0.1, normalization=cast(Normalization, "unknown")).fit(self.x, self.y)
        for tolerance in (-1.0, 0.0, np.inf, np.nan):
            with self.assertRaisesRegex(ValueError, "KKT tolerance.*positive"):
                Lasso(0.1, stopping=StoppingCriterion.kkt_violation(tolerance)).fit(
                    self.x, self.y
                )
        for absolute, relative in (
            (0.0, 0.0),
            (-1.0, 1e-6),
            (1e-6, -1.0),
            (np.nan, 1e-6),
            (0.0, np.inf),
        ):
            with self.assertRaisesRegex(ValueError, "duality-gap tolerances"):
                Lasso(
                    0.1,
                    stopping=StoppingCriterion.duality_gap(relative, absolute=absolute),
                ).fit(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "relative must be zero"):
            Lasso(0.1, stopping=StoppingCriterion("kkt_violation", 1e-6, 1e-6)).fit(
                self.x, self.y
            )
        with self.assertRaisesRegex(ValueError, "stopping kind"):
            Lasso(0.1, stopping=StoppingCriterion(cast(Any, "mapping"), 1e-6)).fit(
                self.x, self.y
            )

    def test_fitting_and_prediction_enforce_dense_input_contract(self) -> None:
        model = Lasso(0.12)
        fit = model.fit(self.x, self.y)
        for x in (self.x.astype(np.float32), self.x.tolist()):
            with self.assertRaises(TypeError):
                model.fit(x, self.y)
            with self.assertRaises(TypeError):
                fit.predict(x)
        with self.assertRaisesRegex(TypeError, "y.*float64"):
            model.fit(self.x, self.y.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            model.fit(self.x, self.y[:, None])
        with self.assertRaisesRegex(ValueError, "y.*length"):
            model.fit(self.x, self.y[:-1])
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            model.fit(np.empty((0, 4)), np.empty(0))
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            fit.predict(self.x[0])
        with self.assertRaisesRegex(ValueError, "columns; expected 4"):
            fit.predict(self.x[:, :3])
        for bad in (np.nan, np.inf, -np.inf):
            x, y = self.x.copy(), self.y.copy()
            x[0, 0] = y[0] = bad
            with self.assertRaisesRegex(ValueError, "X.*finite"):
                model.fit(x, self.y)
            with self.assertRaisesRegex(ValueError, "y.*finite"):
                model.fit(self.x, y)
            with self.assertRaisesRegex(ValueError, "X.*finite"):
                fit.predict(x)
        with self.assertRaisesRegex(ValueError, "copy=False.*Rust-owned"):
            model.fit(self.x, self.y, copy=False)
        with self.assertRaisesRegex(ValueError, "copy=False.*Rust-owned"):
            fit.predict(self.x, copy=False)

    def test_finite_overflow_is_a_floating_point_error(self) -> None:
        with self.assertRaisesRegex(
            FloatingPointError, "nonfinite arithmetic.*iteration 0"
        ):
            Lasso(0.1, normalization="none").fit(
                np.array([[1e308], [-1e308]]), np.array([1.0, -1.0])
            )
        fit = Lasso(0.0, normalization="none").fit(
            np.array([[0.0], [1.0]]), np.array([0.0, 10.0])
        )
        with self.assertRaisesRegex(FloatingPointError, "predicting.*iteration"):
            fit.predict(np.array([[1e308]]))

    def test_native_fit_releases_gil_after_snapshot(self) -> None:
        # A long Python switch interval prevents interpreter scheduling from
        # masquerading as the native solver releasing the GIL.
        rng = np.random.default_rng(42)
        x = rng.normal(size=(10_000, 40))
        y = x @ np.linspace(-2, 2, 40) + rng.normal(size=10_000)
        native = _shrinkage._Lasso(
            0.01, True, "auto", ("kkt_violation", 1e-12, 0.0), 100
        )
        expected = native.fit(x, y)
        ready, start, changed = threading.Event(), threading.Event(), threading.Event()

        def mutate_sources() -> None:
            ready.set()
            start.wait()
            x[:] = 0.0
            y[:] = 0.0
            changed.set()

        worker = threading.Thread(target=mutate_sources)
        worker.start()
        ready.wait()
        interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(10.0)
            start.set()
            actual = native.fit(x, y)
            ran_during_fit = changed.is_set()
        finally:
            sys.setswitchinterval(interval)
            start.set()
            worker.join()
        self.assertTrue(ran_during_fit, "a Python thread could not run during fitting")
        np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
        self.assertEqual(actual.intercept, expected.intercept)


if __name__ == "__main__":
    unittest.main()
