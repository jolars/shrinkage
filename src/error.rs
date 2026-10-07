//! Input, backend, and numerical errors shared by fitting components.

use std::{convert::Infallible, error::Error, fmt};

/// A runtime-selected backend error with its original source preserved.
#[derive(Debug)]
pub struct BackendError(Box<dyn Error>);

impl BackendError {
    /// Erase a concrete backend error while retaining its type and source chain.
    pub fn new<E: Error + 'static>(source: E) -> Self {
        Self(Box::new(source))
    }

    /// Borrow the original error, including for downcasting.
    pub fn as_error(&self) -> &(dyn Error + 'static) {
        self.0.as_ref()
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for BackendError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.as_error())
    }
}

/// Input, numerical, or backend failure. Backend errors retain their type.
///
/// The default source type is [`Infallible`], used by prediction and in-memory
/// matrix backends. Optimization iteration limits are reported in the fitted result.
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
        /// Iteration during which the failure occurred; zero denotes initialization.
        iteration: usize,
    },
    /// A matrix backend failed while computing statistics or a design product.
    Backend {
        /// Operation that requested the backend capability.
        operation: &'static str,
        /// Original backend error.
        source: E,
    },
    /// Backtracking could not establish a valid smooth quadratic upper bound.
    LineSearchFailure {
        /// Iteration whose trial steps were rejected.
        iteration: usize,
        /// Last attempted step size.
        step: f64,
    },
}

impl<E: fmt::Display> fmt::Display for FitError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(f, "invalid fit input: {message}"),
            Self::NumericalFailure {
                operation,
                iteration,
            } => write!(
                f,
                "nonfinite arithmetic while {operation} at iteration {iteration}"
            ),
            Self::Backend { operation, source } => {
                write!(f, "backend failed while {operation}: {source}")
            }
            Self::LineSearchFailure { iteration, step } => write!(
                f,
                "proximal line search failed at iteration {iteration} with step {step}; reduce the initial step or increase max_backtracks"
            ),
        }
    }
}

impl<E> FitError<E> {
    pub(crate) fn erase_backend(self) -> FitError<BackendError>
    where
        E: Error + 'static,
    {
        match self {
            Self::InvalidInput { message } => FitError::InvalidInput { message },
            Self::NumericalFailure {
                operation,
                iteration,
            } => FitError::NumericalFailure {
                operation,
                iteration,
            },
            Self::Backend { operation, source } => FitError::Backend {
                operation,
                source: BackendError::new(source),
            },
            Self::LineSearchFailure { iteration, step } => {
                FitError::LineSearchFailure { iteration, step }
            }
        }
    }

    pub(crate) fn from_component(error: FitError, iteration: usize) -> Self {
        match error {
            FitError::InvalidInput { message } => Self::InvalidInput { message },
            FitError::NumericalFailure { operation, .. } => Self::NumericalFailure {
                operation,
                iteration,
            },
            FitError::LineSearchFailure { iteration, step } => {
                Self::LineSearchFailure { iteration, step }
            }
            FitError::Backend { source, .. } => match source {},
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
