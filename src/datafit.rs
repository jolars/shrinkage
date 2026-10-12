//! Loss capabilities on the linear predictor.

use crate::FitError;
use crate::error::{NumericalFailure, finite, invalid};

/// A Gaussian response with an optional unpenalized fitted intercept.
#[derive(Clone, Copy, Debug)]
pub struct Gaussian<'a> {
    pub(crate) response: &'a [f64],
    pub(crate) fit_intercept: bool,
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

impl SmoothDatafit for Gaussian<'_> {
    fn nobs(&self) -> usize {
        self.response.len()
    }

    fn fits_intercept(&self) -> bool {
        self.fit_intercept
    }

    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.validate(predictor.len())?;
        if output.len() != predictor.len() {
            return Err(invalid("predictor and gradient lengths must match"));
        }
        for ((&value, &target), derivative) in predictor.iter().zip(self.response).zip(output) {
            if !value.is_finite() {
                return Err(invalid("predictor values must be finite"));
            }
            *derivative = finite(
                (value - target) / predictor.len() as f64,
                "computing the predictor gradient",
                0,
            )?;
        }
        Ok(())
    }
}

impl Gaussian<'_> {
    /// Borrow the observed response.
    pub fn response(&self) -> &[f64] {
        self.response
    }

    /// Whether fitting includes a free, unpenalized intercept.
    pub fn fits_intercept(&self) -> bool {
        self.fit_intercept
    }

    pub(crate) fn validate<E>(&self, nrows: usize) -> Result<(), FitError<E>> {
        if nrows == 0 {
            return Err(invalid("training data must have at least one observation"));
        }
        if self.response.len() != nrows {
            return Err(invalid(format!(
                "response length {} does not match {} training rows",
                self.response.len(),
                nrows
            )));
        }
        for (i, value) in self.response.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!("response at row {i} is not finite")));
            }
        }
        Ok(())
    }

    pub(crate) fn residual_loss(
        residual: impl Iterator<Item = f64>,
        n: usize,
        iteration: usize,
    ) -> Result<f64, NumericalFailure> {
        let divisor = (n as f64).sqrt();
        let mut sum = 0.0;
        for residual in residual {
            let value = finite(residual, "reconstructing residuals", iteration)? / divisor;
            sum += 0.5 * value * value;
        }
        finite(sum, "computing the loss", iteration)
    }
}

impl Datafit for Gaussian<'_> {
    type Error = FitError;

    /// Evaluate `||response - predictor||² / (2n)`.
    fn value(&self, predictor: &[f64]) -> Result<f64, Self::Error> {
        self.validate(predictor.len())?;
        if predictor.iter().any(|value| !value.is_finite()) {
            return Err(invalid("predictor values must be finite"));
        }
        Ok(Self::residual_loss(
            self.response.iter().zip(predictor).map(|(y, eta)| y - eta),
            predictor.len(),
            0,
        )?)
    }
}

/// Bernoulli negative log likelihood with a logit link, averaged over rows.
///
/// Responses must be finite and lie in `[0, 1]`; fractional responses represent
/// proportions with equal observation weights. The complete linear predictor
/// includes an optional unpenalized intercept, fitted by default.
///
/// The predictor Hessian is diagonal with entries at most `1 / (4n)`.
/// Consequently, for an optimization-scale design augmented with an intercept
/// column, the coefficient gradient has Lipschitz constant at most
/// `||[X, 1]||² / (4n)`. [`crate::ProximalGradient`] uses backtracking to enforce
/// the corresponding quadratic upper bound without computing this norm.
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::{Col, Mat};
/// use shrinkage::{Binomial, Problem, ProximalGradient, Ridge};
/// let x = Mat::from_fn(4, 1, |i, _| i as f64);
/// let y = [0.0, 0.0, 1.0, 1.0];
/// let fit = Problem::new(&x, Binomial::new(&y), Ridge::new(0.1))
///     .fit_with(&ProximalGradient::<Col<f64>>::new())?;
/// // Predictions are logits; apply the logistic function for probabilities.
/// let logits = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Binomial<'a> {
    response: &'a [f64],
    fit_intercept: bool,
}

impl<'a> Binomial<'a> {
    /// Borrow responses, fitting an intercept by default.
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

    /// Borrow the observed responses.
    pub fn response(&self) -> &[f64] {
        self.response
    }

    /// Whether fitting includes a free, unpenalized intercept.
    pub fn fits_intercept(&self) -> bool {
        self.fit_intercept
    }

    fn validate(&self, predictor: &[f64]) -> Result<(), FitError> {
        if self.response.is_empty() {
            return Err(invalid("training data must have at least one observation"));
        }
        if predictor.len() != self.response.len() {
            return Err(invalid("response and predictor lengths must match"));
        }
        for (i, &target) in self.response.iter().enumerate() {
            if !target.is_finite() || !(0.0..=1.0).contains(&target) {
                return Err(invalid(format!(
                    "binomial response at row {i} must be finite and in [0, 1]"
                )));
            }
        }
        if predictor.iter().any(|value| !value.is_finite()) {
            return Err(invalid("predictor values must be finite"));
        }
        Ok(())
    }
}

impl Datafit for Binomial<'_> {
    type Error = FitError;

    fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
        self.validate(predictor)?;
        let n = predictor.len() as f64;
        let mut loss = 0.0;
        for (&eta, &target) in predictor.iter().zip(self.response) {
            // Avoid subtracting large, nearly equal terms for correctly
            // classified observations, and scale before summing to avoid overflow.
            let linear = if eta >= 0.0 {
                (1.0 - target) * eta
            } else {
                -target * eta
            };
            loss += linear / n + (-eta.abs()).exp().ln_1p() / n;
        }
        Ok(finite(loss, "computing the binomial loss", 0)?)
    }
}

impl SmoothDatafit for Binomial<'_> {
    fn nobs(&self) -> usize {
        self.response.len()
    }

    fn fits_intercept(&self) -> bool {
        self.fit_intercept
    }

    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.validate(predictor)?;
        if output.len() != predictor.len() {
            return Err(invalid("predictor and gradient lengths must match"));
        }
        for ((&eta, &target), derivative) in predictor.iter().zip(self.response).zip(output) {
            let tail = (-eta.abs()).exp();
            // This form retains the small negative derivative when target is
            // one and a large positive logit makes the probability round to one.
            let residual = if eta >= 0.0 {
                (1.0 - target) - tail / (1.0 + tail)
            } else {
                tail / (1.0 + tail) - target
            };
            *derivative = residual / predictor.len() as f64;
        }
        Ok(())
    }
}
