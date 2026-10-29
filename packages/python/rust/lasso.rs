//! Gaussian lasso calls over owned input, detached from the Python interpreter.

use pyo3::exceptions::{PyFloatingPointError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use shrinkage::{
    Centering, FitError, Lasso, LassoFit, Normalization, Scaling, StoppingCriterion, Termination,
};

use crate::input::{DenseDesign, DenseInputs};

#[pyclass(frozen, name = "_Lasso", module = "shrinkage._shrinkage")]
pub(crate) struct PyLasso {
    model: Lasso,
}

#[pymethods]
impl PyLasso {
    #[new]
    fn new(
        lambda_: f64,
        fit_intercept: bool,
        normalization: &str,
        stopping: Option<(String, f64, f64)>,
        max_iterations: isize,
    ) -> PyResult<Self> {
        let normalization = match normalization {
            "auto" => Normalization::Auto,
            "none" => Normalization::None,
            "center" => Normalization::Center,
            "standardize" => Normalization::Standardize,
            "min_max" => Normalization::MinMax,
            "max_abs" => Normalization::MaxAbs,
            "l1" => Normalization::L1,
            "l2" => Normalization::L2,
            _ => {
                return Err(PyValueError::new_err(
                    "normalization must be one of: auto, none, center, standardize, min_max, max_abs, l1, l2",
                ));
            }
        };
        let max_iterations = usize::try_from(max_iterations)
            .map_err(|_| PyValueError::new_err("max_iterations must be strictly positive"))?;
        let mut model = Lasso::new(lambda_)
            .fit_intercept(fit_intercept)
            .normalize(normalization)
            .max_iterations(max_iterations);
        if let Some(stopping) = stopping {
            model = model.terminate_on(parse_stopping(stopping)?);
        }
        Ok(Self { model })
    }

    #[pyo3(signature = (x, y, *, copy = true))]
    fn fit(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<PyLassoFit> {
        let data = DenseInputs::from_python(x, y, copy)?;
        // All buffer exports have ended, and the closure reads only Rust-owned data.
        let fit = py.detach(|| self.model.fit(&data.design.matrix, &data.response));
        fit.map(|fit| PyLassoFit { fit }).map_err(fit_error)
    }
}

fn parse_stopping((kind, absolute, relative): (String, f64, f64)) -> PyResult<StoppingCriterion> {
    match kind.as_str() {
        "duality_gap" => Ok(StoppingCriterion::DualityGap { absolute, relative }),
        "kkt_violation" if relative == 0.0 => Ok(StoppingCriterion::kkt_violation(absolute)),
        "kkt_violation" => Err(PyValueError::new_err(
            "KKT violation uses only an absolute tolerance; relative must be zero",
        )),
        _ => Err(PyValueError::new_err(
            "stopping kind must be duality_gap or kkt_violation for Gaussian lasso",
        )),
    }
}

fn fit_error(error: FitError) -> PyErr {
    let message = error.to_string();
    match error {
        FitError::InvalidInput { .. } | FitError::UnsupportedCombination { .. } => {
            PyValueError::new_err(message)
        }
        FitError::NumericalFailure { .. } => PyFloatingPointError::new_err(message),
        FitError::LineSearchFailure { .. } => PyRuntimeError::new_err(message),
        FitError::Backend { source, .. } => match source {},
    }
}

#[pyclass(frozen, name = "_LassoFit", module = "shrinkage._shrinkage")]
pub(crate) struct PyLassoFit {
    fit: LassoFit,
}

#[pymethods]
impl PyLassoFit {
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.fit.coefficients().to_vec()
    }
    #[getter]
    fn intercept(&self) -> f64 {
        self.fit.intercept()
    }
    #[getter]
    fn centers(&self) -> Option<Vec<f64>> {
        self.fit.preprocessing().centers().map(<[f64]>::to_vec)
    }
    #[getter]
    fn scales(&self) -> Option<Vec<f64>> {
        self.fit.preprocessing().scales().map(<[f64]>::to_vec)
    }
    #[getter]
    fn centering(&self) -> &'static str {
        match self.fit.preprocessing().centering() {
            Centering::None => "none",
            Centering::Mean => "mean",
            Centering::Min => "min",
            Centering::Median => "median",
        }
    }
    #[getter]
    fn scaling(&self) -> &'static str {
        match self.fit.preprocessing().scaling() {
            Scaling::None => "none",
            Scaling::Sd { .. } => "sd",
            Scaling::Range => "range",
            Scaling::MaxAbs => "max_abs",
            Scaling::L1 => "l1",
            Scaling::L2 => "l2",
            Scaling::QuantileRange(_) => "quantile_range",
        }
    }
    #[getter]
    fn termination(&self) -> &'static str {
        match self.fit.termination() {
            Termination::Converged => "converged",
            Termination::IterationLimit => "iteration_limit",
        }
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.fit.iterations()
    }
    #[getter]
    fn objective(&self) -> f64 {
        self.fit.objective()
    }
    #[getter]
    fn kkt_violation(&self) -> f64 {
        self.fit.kkt_violation()
    }
    #[getter]
    fn dual_certificate(&self) -> Vec<f64> {
        self.fit.dual_certificate().to_vec()
    }
    #[getter]
    fn dual_objective(&self) -> f64 {
        self.fit.dual_objective()
    }
    #[getter]
    fn duality_gap(&self) -> f64 {
        self.fit.duality_gap()
    }
    #[getter]
    fn reference_loss(&self) -> f64 {
        self.fit.reference_loss()
    }
    #[getter]
    fn stopping(&self) -> (&'static str, f64, f64) {
        match self.fit.stopping_criterion() {
            StoppingCriterion::DualityGap { absolute, relative } => {
                ("duality_gap", absolute, relative)
            }
            StoppingCriterion::KktViolation { absolute } => ("kkt_violation", absolute, 0.0),
            StoppingCriterion::ProximalGradientMapping { .. } => {
                unreachable!("lasso rejects mapping criteria")
            }
        }
    }
    #[getter]
    fn stopping_value(&self) -> f64 {
        self.fit.stopping_value()
    }
    #[getter]
    fn stopping_threshold(&self) -> f64 {
        self.fit.stopping_threshold()
    }

    #[pyo3(signature = (x, *, copy = true))]
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>, copy: bool) -> PyResult<Vec<f64>> {
        let data = DenseDesign::from_python(x, copy)?;
        py.detach(|| self.fit.predict(&data.matrix))
            .map_err(fit_error)
    }
}
