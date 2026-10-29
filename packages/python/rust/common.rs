//! Shared option conversion and owned errors across the interpreter boundary.

use std::fmt::Display;

use pyo3::exceptions::{PyFloatingPointError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use shrinkage::{Centering, FitError, Normalization, Scaling, StoppingCriterion, Termination};

pub(crate) fn normalization(value: &str) -> PyResult<Normalization> {
    match value {
        "auto" => Ok(Normalization::Auto),
        "none" => Ok(Normalization::None),
        "center" => Ok(Normalization::Center),
        "standardize" => Ok(Normalization::Standardize),
        "min_max" => Ok(Normalization::MinMax),
        "max_abs" => Ok(Normalization::MaxAbs),
        "l1" => Ok(Normalization::L1),
        "l2" => Ok(Normalization::L2),
        _ => Err(PyValueError::new_err(
            "normalization must be one of: auto, none, center, standardize, min_max, max_abs, l1, l2",
        )),
    }
}

pub(crate) fn parse_stopping(
    (kind, absolute, relative): (String, f64, f64),
) -> PyResult<StoppingCriterion> {
    match kind.as_str() {
        "duality_gap" => Ok(StoppingCriterion::DualityGap { absolute, relative }),
        "kkt_violation" if relative == 0.0 => Ok(StoppingCriterion::kkt_violation(absolute)),
        "proximal_gradient_mapping" if relative == 0.0 => {
            Ok(StoppingCriterion::proximal_gradient_mapping(absolute))
        }
        "kkt_violation" | "proximal_gradient_mapping" => Err(PyValueError::new_err(
            "KKT violation and proximal-gradient mapping use only an absolute tolerance; relative must be zero",
        )),
        _ => Err(PyValueError::new_err(
            "stopping kind must be duality_gap, kkt_violation, or proximal_gradient_mapping",
        )),
    }
}

pub(crate) fn stopping(criterion: StoppingCriterion) -> (&'static str, f64, f64) {
    match criterion {
        StoppingCriterion::DualityGap { absolute, relative } => ("duality_gap", absolute, relative),
        StoppingCriterion::KktViolation { absolute } => ("kkt_violation", absolute, 0.0),
        StoppingCriterion::ProximalGradientMapping { absolute } => {
            ("proximal_gradient_mapping", absolute, 0.0)
        }
    }
}

pub(crate) fn centering(value: Centering) -> &'static str {
    match value {
        Centering::None => "none",
        Centering::Mean => "mean",
        Centering::Min => "min",
        Centering::Median => "median",
    }
}

pub(crate) fn scaling(value: Scaling) -> &'static str {
    match value {
        Scaling::None => "none",
        Scaling::Sd { .. } => "sd",
        Scaling::Range => "range",
        Scaling::MaxAbs => "max_abs",
        Scaling::L1 => "l1",
        Scaling::L2 => "l2",
        Scaling::QuantileRange(_) => "quantile_range",
    }
}

pub(crate) fn termination(value: Termination) -> &'static str {
    match value {
        Termination::Converged => "converged",
        Termination::IterationLimit => "iteration_limit",
    }
}

pub(crate) fn positive(name: &str, value: isize) -> PyResult<usize> {
    if value <= 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be strictly positive"
        )));
    }
    Ok(value as usize)
}

pub(crate) fn nonnegative(name: &str, value: isize) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| PyValueError::new_err(format!("{name} must be nonnegative")))
}

enum ErrorKind {
    Value,
    FloatingPoint,
    Runtime,
}

/// Runtime backend sources need not be Send, so transport only their owned message.
pub(crate) struct DetachedError {
    kind: ErrorKind,
    message: String,
}

impl<E: Display> From<FitError<E>> for DetachedError {
    fn from(error: FitError<E>) -> Self {
        let message = error.to_string();
        let kind = match error {
            FitError::InvalidInput { .. } | FitError::UnsupportedCombination { .. } => {
                ErrorKind::Value
            }
            FitError::NumericalFailure { .. } => ErrorKind::FloatingPoint,
            FitError::LineSearchFailure { .. } | FitError::Backend { .. } => ErrorKind::Runtime,
        };
        Self { kind, message }
    }
}

impl DetachedError {
    pub(crate) fn into_python(self) -> PyErr {
        match self.kind {
            ErrorKind::Value => PyValueError::new_err(self.message),
            ErrorKind::FloatingPoint => PyFloatingPointError::new_err(self.message),
            ErrorKind::Runtime => PyRuntimeError::new_err(self.message),
        }
    }
}

pub(crate) fn fit_error<E: Display>(error: FitError<E>) -> PyErr {
    DetachedError::from(error).into_python()
}
