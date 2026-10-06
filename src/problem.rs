//! Typed components for the Gaussian lasso coordinate solver.

use lazymatrix::{ColumnStats, RawColumns};

use crate::{Lasso, LassoError, LassoFit, Normalization};

/// A Gaussian response with an optional unpenalized fitted intercept.
#[derive(Clone, Copy, Debug)]
pub struct Gaussian<'a> {
    response: &'a [f64],
    fit_intercept: bool,
}

impl<'a> Gaussian<'a> {
    /// Borrow a response, fitting an intercept by default.
    pub fn new(response: &'a [f64]) -> Self {
        Self {
            response,
            fit_intercept: true,
        }
    }

    /// Enable or disable the unpenalized fitted intercept.
    ///
    /// Explicit column centering can still induce an original-scale intercept.
    pub fn fit_intercept(mut self, enabled: bool) -> Self {
        self.fit_intercept = enabled;
        self
    }
}

/// An L1 penalty on optimization-scale coefficients.
#[derive(Clone, Copy, Debug)]
pub struct L1 {
    lambda: f64,
}

impl L1 {
    /// Set the penalty strength, checked when fitting.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

/// Cyclic coordinate descent settings for Gaussian lasso.
#[derive(Clone, Copy, Debug)]
pub struct CoordinateDescent {
    tolerance: f64,
    max_iterations: usize,
}

impl CoordinateDescent {
    /// Use an absolute KKT tolerance of `1e-6` and at most 10,000 sweeps.
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 10_000,
        }
    }

    /// Set the absolute KKT tolerance, checked when fitting.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum number of complete sweeps, checked when fitting.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

impl Default for CoordinateDescent {
    fn default() -> Self {
        Self::new()
    }
}

/// A typed design, datafit, and penalty with training normalization settings.
///
/// The implemented combination is a borrowed [`RawColumns`] design,
/// [`Gaussian`] response, and [`L1`] penalty, fitted by [`CoordinateDescent`].
/// Its result is an owned [`LassoFit`] with original-scale parameters.
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::Mat;
/// use shrinkage::{CoordinateDescent, Gaussian, L1, Normalization, Problem};
///
/// let x = Mat::from_fn(3, 1, |i, _| i as f64);
/// let y = [1.0, 3.0, 5.0];
/// let fit = Problem::new(&x, Gaussian::new(&y), L1::new(0.1))
///     .normalize(Normalization::Standardize)
///     .fit_with(&CoordinateDescent::new())?;
/// let predictions = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Problem<X, D, P> {
    design: X,
    datafit: D,
    penalty: P,
    normalization: Normalization,
    supplied_centers: Option<Vec<f64>>,
    supplied_scales: Option<Vec<f64>>,
}

impl<X, D, P> Problem<X, D, P> {
    /// Compose a design, datafit, and penalty.
    pub fn new(design: X, datafit: D, penalty: P) -> Self {
        Self {
            design,
            datafit,
            penalty,
            normalization: Normalization::Auto,
            supplied_centers: None,
            supplied_scales: None,
        }
    }

    /// Select lazy training-column normalization.
    pub fn normalize(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Supply centers in training-column order, checked when fitting.
    pub fn with_centers(mut self, centers: Vec<f64>) -> Self {
        self.supplied_centers = Some(centers);
        self
    }

    /// Supply positive scales in training-column order, checked when fitting.
    pub fn with_scales(mut self, scales: Vec<f64>) -> Self {
        self.supplied_scales = Some(scales);
        self
    }
}

impl<M> Problem<&M, Gaussian<'_>, L1>
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
{
    /// Fit Gaussian lasso by coordinate descent.
    ///
    /// Penalties act on normalized coefficients. Returned coefficients,
    /// intercept, and predictions use the original design scale, including a
    /// centering-induced intercept when fitted intercepts are disabled.
    ///
    /// # Errors
    /// Returns [`LassoError`] for invalid input, nonfinite arithmetic, or
    /// backend preprocessing failure. An iteration limit returns a finite fit.
    pub fn fit_with(&self, solver: &CoordinateDescent) -> Result<LassoFit, LassoError<M::Error>> {
        let mut model = Lasso::new(self.penalty.lambda)
            .fit_intercept(self.datafit.fit_intercept)
            .normalize(self.normalization)
            .tolerance(solver.tolerance)
            .max_iterations(solver.max_iterations);
        if let Some(centers) = &self.supplied_centers {
            model = model.with_centers(centers.clone());
        }
        if let Some(scales) = &self.supplied_scales {
            model = model.with_scales(scales.clone());
        }
        model.fit(self.design, self.datafit.response)
    }
}
