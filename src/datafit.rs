//! Loss capabilities on the linear predictor.

mod binomial;
mod gaussian;
mod linked;
mod multivariate;
pub use binomial::Binomial;
pub use gaussian::Gaussian;
pub use linked::LinkedDatafit;
pub use multivariate::MultivariateGaussian;
pub(crate) use multivariate::StackedGaussian;

use crate::FitError;

/// Datafit `F` evaluated on a complete linear predictor, including any intercept.
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

/// A differentiable loss on the complete linear predictor.
///
/// Proximal gradient uses this derivative with forward and transposed design
/// products. Implementations must use the same normalization in both methods.
pub trait DifferentiableDatafit: Datafit<Error = FitError> {
    /// Number of training observations.
    fn n_obs(&self) -> usize;

    /// Whether the solver should fit a free, unpenalized intercept.
    fn fits_intercept(&self) -> bool;

    /// Write the predictor gradient into reusable output storage.
    ///
    /// # Errors
    /// Reject invalid response data, dimensions, predictor values, or arithmetic.
    /// Output is unspecified after an error.
    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError>;
}

/// A differentiable convex loss, optionally providing a dual certificate.
pub trait SmoothDatafit: DifferentiableDatafit {
    /// Borrow an optional original-loss dual certificate capability.
    fn gap_safe(&self) -> Option<&dyn crate::GapSafeDatafit> {
        None
    }
}

/// A twice-differentiable loss whose predictor Hessian is diagonal.
///
/// Off-diagonal Hessian entries must be zero. Entries use exactly the same
/// normalization as [`DifferentiableDatafit::gradient`], including row averaging.
/// Entries may be negative; this capability does not imply convexity or provide
/// the positive working curvature required by [`crate::Irls`].
pub trait DiagonalHessianDatafit: DifferentiableDatafit {
    /// Write finite, signed predictor Hessian entries into reusable storage.
    ///
    /// # Errors
    /// Reject invalid data, predictor values, dimensions, or arithmetic.
    /// Output is unspecified after an error.
    fn hessian_diagonal(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError>;
}

/// A differentiable loss with nonnegative diagonal working curvature.
///
/// Curvature constructs a local quadratic model; it need not be the observed
/// Hessian. IRLS checks descent and stationarity against the original loss.
pub trait QuadraticModelDatafit: DifferentiableDatafit {
    /// Write finite, nonnegative curvature using the loss's normalization.
    ///
    /// # Errors
    /// Reject invalid inputs or arithmetic. Output is unspecified after failure.
    fn working_curvature(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError>;

    /// Choose a finite cold-start intercept, leaving existing models at zero.
    ///
    /// # Errors
    /// Reject invalid responses, link configuration, or initialization.
    fn initial_intercept(&self) -> Result<f64, FitError> {
        Ok(0.0)
    }

    /// Whether a predictor lies in the domain of the original loss.
    fn valid_predictor(&self, _predictor: &[f64]) -> bool {
        true
    }
}

impl<D: DiagonalHessianDatafit + ?Sized> DiagonalHessianDatafit for Box<D> {
    fn hessian_diagonal(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        (**self).hessian_diagonal(predictor, output)
    }
}

impl<D: QuadraticModelDatafit + ?Sized> QuadraticModelDatafit for Box<D> {
    fn working_curvature(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        (**self).working_curvature(predictor, output)
    }
    fn initial_intercept(&self) -> Result<f64, FitError> {
        (**self).initial_intercept()
    }
    fn valid_predictor(&self, predictor: &[f64]) -> bool {
        (**self).valid_predictor(predictor)
    }
}
