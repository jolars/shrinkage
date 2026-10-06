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
