//! Gaussian lasso convenience configuration.

use crate::{CoordinateDescent, Gaussian, L1, Normalization, Problem};
pub use crate::{
    error::FitError as LassoError,
    fit::{LassoFit, Preprocessing, Termination},
    solver::StoppingCriterion,
};
use lazymatrix::{ColumnStats, RawColumns};

/// Gaussian lasso configuration.
///
/// Fits `||y - X_tilde * theta - intercept||² / (2n) + lambda * ||theta||₁`.
/// The intercept is unpenalized. By default, `X_tilde` uses training-column
/// means and population standard deviations, and the penalty acts on the
/// normalized coefficients. Reported coefficients and predictions use the
/// original input scale.
///
/// Dense and CSC matrices use the same statically dispatched iteration routine.
/// Solver storage is `O(n + p)`; neither a normalized matrix nor a Gram matrix
/// is materialized. Options are validated when [`Self::fit`] is called.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::Mat;
/// use shrinkage::{Lasso, Termination};
///
/// let x = Mat::from_fn(3, 1, |i, _| i as f64);
/// let fit = Lasso::new(0.1).fit(&x, &[1.0, 3.0, 5.0])?;
/// assert_eq!(fit.termination(), Termination::Converged);
/// let predictions = fit.predict(&x)?;
/// # }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct Lasso {
    lambda: f64,
    fit_intercept: bool,
    normalization: Normalization,
    supplied_centers: Option<Vec<f64>>,
    supplied_scales: Option<Vec<f64>>,
    stopping_criterion: StoppingCriterion,
    max_iterations: usize,
}

impl Lasso {
    /// Configure a fit with a finite, nonnegative penalty strength.
    ///
    /// Defaults to an intercept, [`Normalization::Auto`], a relative duality
    /// gap tolerance of `1e-6`, and at most `10_000` complete coordinate sweeps.
    /// At zero penalty, the default is absolute KKT tolerance `1e-6`: scaling
    /// a residual cannot generally construct a useful zero-penalty dual vector.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            fit_intercept: true,
            normalization: Normalization::Auto,
            supplied_centers: None,
            supplied_scales: None,
            stopping_criterion: if lambda == 0.0 {
                StoppingCriterion::kkt_violation(1e-6)
            } else {
                StoppingCriterion::duality_gap(1e-6)
            },
            max_iterations: 10_000,
        }
    }

    /// Enable or disable the unpenalized intercept.
    ///
    /// With [`Normalization::Auto`], disabling the intercept also disables
    /// centering while retaining population-standard-deviation scaling.
    /// Explicit normalization choices are honored independently. Explicit
    /// centering can induce a fixed original-scale intercept even when this
    /// option is disabled.
    pub fn fit_intercept(mut self, enabled: bool) -> Self {
        self.fit_intercept = enabled;
        self
    }

    /// Select a training-column normalization preset or custom specification.
    ///
    /// Defaults to [`Normalization::Auto`]. Use [`Normalization::None`] for
    /// the raw design or [`Normalization::Custom`] to choose centering and
    /// scaling independently. Computed zero scales are replaced with one.
    /// Penalties act on normalized coefficients; reported parameters and
    /// predictions use the original scale.
    pub fn normalize(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Supply column centers instead of computing them from the training data.
    ///
    /// This overrides the centering rule selected by [`Self::normalize`]. Each
    /// center must be finite, and the vector must have one entry per column.
    /// The values are checked by [`Self::fit`]. If scales are computed using
    /// L1, L2, or maximum-absolute scaling, they use these centers.
    pub fn with_centers(mut self, centers: Vec<f64>) -> Self {
        self.supplied_centers = Some(centers);
        self
    }

    /// Supply column scales instead of computing them from the training data.
    ///
    /// This overrides the scaling rule selected by [`Self::normalize`]. Each
    /// scale must be finite and strictly positive, and the vector must have one
    /// entry per column. The values are checked by [`Self::fit`].
    pub fn with_scales(mut self, scales: Vec<f64>) -> Self {
        self.supplied_scales = Some(scales);
        self
    }

    /// Set a finite, strictly positive absolute KKT tolerance.
    ///
    /// With `c_j = X_tilde_j' * residual / n`, the violation is
    /// `|c_j - lambda * sign(theta_j)|` for nonzero coefficients and
    /// `max(|c_j| - lambda, 0)` otherwise. When fitting an intercept, include
    /// `|mean(residual)|`. Convergence requires the largest violation to be at
    /// most this tolerance, confirmed after reconstructing the residual.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.stopping_criterion = StoppingCriterion::kkt_violation(tolerance);
        self
    }

    /// Select the stopping criterion and its tolerances.
    ///
    /// The duality gap uses a fixed reference loss from the zero-coefficient
    /// model, with the intercept optimized when enabled. The legacy
    /// [`Self::tolerance`] setter selects absolute KKT violation.
    pub fn terminate_on(mut self, criterion: StoppingCriterion) -> Self {
        self.stopping_criterion = criterion;
        self
    }

    /// Set a strictly positive limit on complete coordinate sweeps.
    ///
    /// An initially optimal fit takes zero sweeps. Reaching the limit returns
    /// a finite fit with [`Termination::IterationLimit`], not convergence.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Fit against raw training columns and a response slice.
    ///
    /// Zero-feature designs are supported. Zero-norm normalized columns stay
    /// at zero. The fitted result owns its parameters and does not borrow `x`.
    ///
    /// # Errors
    /// Returns [`LassoError::InvalidInput`] for invalid dimensions, empty
    /// observations, nonfinite inputs, or invalid options. A preprocessing
    /// failure retains its backend source in [`LassoError::Backend`]. Finite
    /// inputs whose arithmetic produces nonfinite values return
    /// [`LassoError::NumericalFailure`]. Iteration limits return a fit instead.
    pub fn fit<M>(&self, x: &M, y: &[f64]) -> Result<LassoFit, LassoError<M::Error>>
    where
        M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
    {
        let mut problem = Problem::new(
            x,
            Gaussian::new(y).fit_intercept(self.fit_intercept),
            L1::new(self.lambda),
        )
        .normalize(self.normalization);
        if let Some(centers) = &self.supplied_centers {
            problem = problem.with_centers(centers.clone());
        }
        if let Some(scales) = &self.supplied_scales {
            problem = problem.with_scales(scales.clone());
        }
        problem.fit_with(
            &CoordinateDescent::new()
                .terminate_on(self.stopping_criterion)
                .max_iterations(self.max_iterations),
        )
    }
}
