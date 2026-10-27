//! Loss capabilities on the linear predictor.

mod binomial;
mod gaussian;
mod multivariate;
pub use binomial::Binomial;
pub use gaussian::Gaussian;
pub use multivariate::MultivariateGaussian;
pub(crate) use multivariate::StackedGaussian;

use crate::FitError;

/// Loss evaluated on a complete linear predictor, including any intercept.
///
/// Derivatives and solver-specific updates are separate capabilities. Matrix
/// products belong to the design, not the datafit.
pub trait Datafit {
    /// Failure returned by loss evaluation.
    type Error;

    /// Evaluate the loss using this datafit's normalization convention.
    ///
    /// # Errors
    /// Returns an error for invalid data, predictor dimensions, or arithmetic.
    fn value(&self, predictor: &[f64]) -> Result<f64, Self::Error>;
}

/// A differentiable convex loss on the complete linear predictor.
///
/// Proximal gradient uses this derivative with forward and transposed design
/// products. Implementations must use the same normalization in both methods.
pub trait SmoothDatafit: Datafit<Error = FitError> {
    /// Number of training observations.
    fn nobs(&self) -> usize;

    /// Whether the solver should fit a free, unpenalized intercept.
    fn fits_intercept(&self) -> bool;

    /// Write the predictor gradient into reusable output storage.
    ///
    /// # Errors
    /// Reject invalid response data, dimensions, predictor values, or arithmetic.
    /// Output is unspecified after an error.
    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError>;
}

/// A convex smooth loss whose predictor Hessian is diagonal.
///
/// Off-diagonal Hessian entries must be zero. Entries use exactly the same
/// normalization as [`SmoothDatafit::gradient`], including any row averaging.
/// This excludes losses that couple observations. Zero curvature is allowed;
/// [`crate::Irls`] stabilizes its quadratic model without changing the loss.
pub trait DiagonalHessianDatafit: SmoothDatafit {
    /// Write finite, nonnegative predictor Hessian entries into reusable storage.
    ///
    /// # Errors
    /// Reject invalid data, predictor values, dimensions, or arithmetic.
    /// Output is unspecified after an error.
    fn hessian_diagonal(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError>;
}

impl<D: DiagonalHessianDatafit + ?Sized> DiagonalHessianDatafit for Box<D> {
    fn hessian_diagonal(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        (**self).hessian_diagonal(predictor, output)
    }
}
