//! Gaussian lasso with cyclic coordinate descent and lazy normalization.

mod solver;

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use lazymatrix::{ColumnStats, LazyMatrix, RawColumn, RawColumns};

use crate::{Centering, Normalization, Scaling};
use solver::{NumericalFailure, finite};

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
    tolerance: f64,
    max_iterations: usize,
}

impl Lasso {
    /// Configure a fit with a finite, nonnegative penalty strength.
    ///
    /// Defaults to an intercept, [`Normalization::Auto`], an absolute KKT
    /// tolerance of `1e-6`, and at most `10_000` complete coordinate sweeps.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            fit_intercept: true,
            normalization: Normalization::Auto,
            tolerance: 1e-6,
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

    /// Set a finite, strictly positive absolute KKT tolerance.
    ///
    /// With `c_j = X_tilde_j' * residual / n`, the violation is
    /// `|c_j - lambda * sign(theta_j)|` for nonzero coefficients and
    /// `max(|c_j| - lambda, 0)` otherwise. When fitting an intercept, include
    /// `|mean(residual)|`. Convergence requires the largest violation to be at
    /// most this tolerance, confirmed after reconstructing the residual.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
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
        self.validate()?;
        if x.nrows() == 0 {
            return Err(invalid("training data must have at least one observation"));
        }
        if y.len() != x.nrows() {
            return Err(invalid(format!(
                "response length {} does not match {} training rows",
                y.len(),
                x.nrows()
            )));
        }
        for (i, value) in y.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!("response at row {i} is not finite")));
            }
        }
        validate_matrix(x)?;
        let spec = self.normalization.specification(self.fit_intercept);
        let (centers, mut scales) = if spec.center != Centering::None || spec.scale != Scaling::None
        {
            x.normalization_stats(spec)
                .map_err(|source| LassoError::Backend {
                    operation: "computing training normalization statistics",
                    source,
                })?
        } else {
            (None, None)
        };
        validate_statistics(
            &centers,
            x.ncols(),
            spec.center != Centering::None,
            "column centers",
        )?;
        validate_statistics(
            &scales,
            x.ncols(),
            spec.scale != Scaling::None,
            "column scales",
        )?;
        if let Some(scales) = &mut scales {
            for scale in scales {
                if *scale < 0.0 {
                    return Err(invalid(
                        "matrix backend returned a negative normalization scale",
                    ));
                }
                if *scale == 0.0 {
                    *scale = 1.0;
                }
            }
        }
        let matrix = LazyMatrix::from_parts(x, centers, scales);
        let solution = solver::solve(&matrix, y, self)?;
        let (_, centers, scales) = matrix.into_parts();
        let mut coefficients = solution.coefficients;
        if let Some(scales) = &scales {
            for (coefficient, scale) in coefficients.iter_mut().zip(scales) {
                *coefficient = finite(
                    *coefficient / scale,
                    "back-transforming coefficients",
                    solution.iterations,
                )?;
            }
        }
        let mut intercept = solution.intercept;
        if let Some(centers) = &centers {
            for (coefficient, center) in coefficients.iter().zip(centers) {
                intercept = finite(
                    intercept - coefficient * center,
                    "back-transforming intercept",
                    solution.iterations,
                )?;
            }
        }
        Ok(LassoFit {
            coefficients,
            intercept,
            preprocessing: Preprocessing {
                spec,
                centers,
                scales,
            },
            termination: solution.termination,
            iterations: solution.iterations,
            objective: solution.objective,
            kkt_violation: solution.kkt_violation,
        })
    }

    fn validate<E>(&self) -> Result<(), LassoError<E>> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(invalid("lambda must be finite and nonnegative"));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(invalid("tolerance must be finite and strictly positive"));
        }
        if self.max_iterations == 0 {
            return Err(invalid("max_iterations must be strictly positive"));
        }
        Ok(())
    }
}

/// Training normalization retained by a fitted lasso.
#[derive(Clone, Debug)]
pub struct Preprocessing {
    spec: lazymatrix::Normalization,
    centers: Option<Vec<f64>>,
    scales: Option<Vec<f64>>,
}

impl Preprocessing {
    /// The centering rule used during training, with automatic choices resolved.
    pub fn centering(&self) -> Centering {
        self.spec.center
    }

    /// The scaling rule used during training, with automatic choices resolved.
    pub fn scaling(&self) -> Scaling {
        self.spec.scale
    }

    /// Training-column centers, or `None` when centering was disabled.
    pub fn centers(&self) -> Option<&[f64]> {
        self.centers.as_deref()
    }

    /// Training-column scales with zeros replaced by one, or `None` when disabled.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }
}

/// Why a finite fit stopped. An iteration limit does not certify optimality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The full KKT check passed on a freshly reconstructed residual.
    Converged,
    /// The sweep budget was exhausted before meeting the KKT tolerance.
    IterationLimit,
}

/// Owned original-scale parameters, preprocessing metadata, and diagnostics.
///
/// Check [`Self::termination`] before treating a fit as converged. Diagnostics
/// refer to the optimization-scale penalty even though coefficients use the
/// original scale.
#[derive(Clone, Debug)]
#[must_use]
pub struct LassoFit {
    coefficients: Vec<f64>,
    intercept: f64,
    preprocessing: Preprocessing,
    termination: Termination,
    iterations: usize,
    objective: f64,
    kkt_violation: f64,
}

impl LassoFit {
    /// Original-scale coefficients, in training-column order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Original-scale intercept, including the correction for centering.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Fitted training-column centers and scales.
    pub fn preprocessing(&self) -> &Preprocessing {
        &self.preprocessing
    }

    /// Whether the fit converged or exhausted its iteration budget.
    pub fn termination(&self) -> Termination {
        self.termination
    }

    /// Number of complete coordinate sweeps performed.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Final `||residual||² / (2n) + lambda * ||theta||₁`.
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// Maximum absolute KKT violation, as defined by [`Lasso::tolerance`].
    pub fn kkt_violation(&self) -> f64 {
        self.kkt_violation
    }

    /// Predict from raw columns in the same order as the training design.
    ///
    /// Uses original-scale parameters, without estimating new centers or
    /// scales. Empty prediction batches return an empty vector.
    ///
    /// # Errors
    /// Returns an error for a feature-count mismatch, nonfinite input, or
    /// nonfinite prediction arithmetic.
    pub fn predict<M: RawColumns<f64> + ?Sized>(&self, x: &M) -> Result<Vec<f64>, LassoError> {
        if x.ncols() != self.coefficients.len() {
            return Err(invalid(format!(
                "prediction design has {} columns; expected {}",
                x.ncols(),
                self.coefficients.len()
            )));
        }
        validate_matrix(x)?;
        let mut predictions = vec![self.intercept; x.nrows()];
        for (j, &coefficient) in self.coefficients.iter().enumerate() {
            if coefficient != 0.0 {
                x.raw_column(j)
                    .affine_add_to(coefficient, 0.0, &mut predictions);
            }
        }
        for &value in &predictions {
            finite(value, "predicting", self.iterations)?;
        }
        Ok(predictions)
    }
}

/// Input, numerical, or backend failure. Backend errors retain their type.
///
/// The default source type is [`Infallible`], used by prediction and in-memory
/// matrix backends. Optimization iteration limits are reported in [`LassoFit`].
#[derive(Debug)]
pub enum LassoError<E = Infallible> {
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

impl<E: fmt::Display> fmt::Display for LassoError<E> {
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

impl<E: Error + 'static> Error for LassoError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl<E> From<NumericalFailure> for LassoError<E> {
    fn from(error: NumericalFailure) -> Self {
        Self::NumericalFailure {
            operation: error.operation,
            iteration: error.iteration,
        }
    }
}

fn invalid<E>(message: impl Into<String>) -> LassoError<E> {
    LassoError::InvalidInput {
        message: message.into(),
    }
}

fn validate_matrix<M: RawColumns<f64> + ?Sized, E>(x: &M) -> Result<(), LassoError<E>> {
    for j in 0..x.ncols() {
        let column = x.raw_column(j);
        if column.len() != x.nrows() || column.stored_len() > x.nrows() {
            return Err(invalid(format!(
                "matrix backend returned invalid dimensions for column {j}"
            )));
        }
        let mut invalid_row = None;
        column.for_each_stored(|i, value| {
            if (i >= x.nrows() || !value.is_finite()) && invalid_row.is_none() {
                invalid_row = Some(i);
            }
        });
        if let Some(i) = invalid_row {
            return Err(invalid(format!(
                "invalid matrix entry at row {i}, column {j}: expected an in-bounds, finite value"
            )));
        }
    }
    Ok(())
}

fn validate_statistics<E>(
    values: &Option<Vec<f64>>,
    ncols: usize,
    expected: bool,
    operation: &'static str,
) -> Result<(), LassoError<E>> {
    if values.is_some() != expected || values.as_ref().is_some_and(|v| v.len() != ncols) {
        return Err(invalid(format!(
            "matrix backend returned unexpected {operation}"
        )));
    }
    if let Some(values) = values {
        for &value in values {
            finite(value, operation, 0)?;
        }
    }
    Ok(())
}
