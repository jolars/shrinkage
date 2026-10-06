//! Input, backend, and numerical errors shared by fitting components.

use std::{convert::Infallible, error::Error, fmt};

/// Input, numerical, or backend failure. Backend errors retain their type.
///
/// The default source type is [`Infallible`], used by prediction and in-memory
/// matrix backends. Optimization iteration limits are reported in [`crate::LassoFit`].
#[derive(Debug)]
pub enum FitError<E = Infallible> {
    /// Invalid data, dimensions, options, or backend-provided statistics.
    InvalidInput {
        /// Description of the invalid input, including its location when available.
        message: String,
    },
    /// Arithmetic produced a nonfinite value from finite inputs.
    NumericalFailure {
        /// Operation that encountered the nonfinite value.
        operation: &'static str,
        /// Sweep during which the failure occurred; zero denotes initialization.
        iteration: usize,
    },
    /// A matrix backend failed while computing preprocessing statistics.
    Backend {
        /// Operation that requested the backend capability.
        operation: &'static str,
        /// Original backend error.
        source: E,
    },
}

impl<E: fmt::Display> fmt::Display for FitError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(f, "invalid lasso input: {message}"),
            Self::NumericalFailure {
                operation,
                iteration,
            } => write!(
                f,
                "nonfinite arithmetic while {operation} at sweep {iteration}"
            ),
            Self::Backend { operation, source } => {
                write!(f, "backend failed while {operation}: {source}")
            }
        }
    }
}

impl<E: Error + 'static> Error for FitError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl<E> From<NumericalFailure> for FitError<E> {
    fn from(error: NumericalFailure) -> Self {
        Self::NumericalFailure {
            operation: error.operation,
            iteration: error.iteration,
        }
    }
}

pub(crate) fn invalid<E>(message: impl Into<String>) -> FitError<E> {
    FitError::InvalidInput {
        message: message.into(),
    }
}

#[derive(Debug)]
pub(crate) struct NumericalFailure {
    pub operation: &'static str,
    pub iteration: usize,
}

pub(crate) fn finite(
    value: f64,
    operation: &'static str,
    iteration: usize,
) -> Result<f64, NumericalFailure> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NumericalFailure {
            operation,
            iteration,
        })
    }
}
