use std::{convert::Infallible, error::Error, fmt};

use crate::FitError;

/// Validation or fitting failure with the fold and candidate when available.
///
/// Errors return no partial cross-validation result. A nonconverged fit is an
/// error, so an incomplete candidate cannot silently influence selection.
#[derive(Debug)]
pub enum CrossValidationError<E = Infallible> {
    /// Invalid configuration, folds, or full response.
    InvalidInput {
        /// Underlying validation failure.
        source: FitError<E>,
    },
    /// Training, path fitting, or prediction failed in a validation fold.
    Fold {
        /// Zero-based held-out fold identifier.
        fold: usize,
        /// Original validation, backend, or numerical failure.
        source: FitError<E>,
    },
    /// A path point did not meet its solver's convergence criterion.
    NonConverged {
        /// Held-out fold, or `None` for the full-data refit path.
        fold: Option<usize>,
        /// Zero-based candidate position in path order.
        candidate: usize,
    },
    /// Fitting the full-data path failed after selection.
    Refit {
        /// Original fitting failure.
        source: FitError<E>,
    },
}

impl<E> From<FitError<E>> for CrossValidationError<E> {
    fn from(source: FitError<E>) -> Self {
        Self::InvalidInput { source }
    }
}

impl<E: fmt::Display> fmt::Display for CrossValidationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { source } => write!(f, "invalid cross-validation input: {source}"),
            Self::Fold { fold, source } => {
                write!(f, "cross-validation fold {fold} failed: {source}")
            }
            Self::NonConverged {
                fold: Some(fold),
                candidate,
            } => write!(
                f,
                "cross-validation fold {fold}, candidate {candidate} did not converge"
            ),
            Self::NonConverged {
                fold: None,
                candidate,
            } => write!(f, "full-data refit candidate {candidate} did not converge"),
            Self::Refit { source } => {
                write!(f, "cross-validation full-data refit failed: {source}")
            }
        }
    }
}

impl<E: Error + 'static> Error for CrossValidationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidInput { source } | Self::Fold { source, .. } | Self::Refit { source } => {
                Some(source)
            }
            Self::NonConverged { .. } => None,
        }
    }
}
