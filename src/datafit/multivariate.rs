//! Gaussian loss with explicit observation and response dimensions.

use crate::error::{finite, invalid};
use crate::{Datafit, FitError, Gaussian, SmoothDatafit};

/// Gaussian loss `||Y - predictor||_F² / (2 * nobs)`.
///
/// Responses and predictors store each response column contiguously: entry
/// `(observation, response)` has index `response * nobs + observation`. The
/// loss sums over responses and averages over observations, so separable fits
/// use the same penalty strengths as independent [`Gaussian`] fits. Each
/// response has its own optional unpenalized intercept, enabled by default.
#[derive(Clone, Copy, Debug)]
pub struct MultivariateGaussian<'a> {
    response: &'a [f64],
    nobs: usize,
    nresponses: usize,
    fit_intercept: bool,
}

impl<'a> MultivariateGaussian<'a> {
    /// Borrow response-major observations with explicit dimensions.
    /// Dimensions and entries are checked during loss evaluation or fitting.
    pub fn new(response: &'a [f64], nobs: usize, nresponses: usize) -> Self {
        Self {
            response,
            nobs,
            nresponses,
            fit_intercept: true,
        }
    }

    /// Enable or disable the free intercept for every response.
    /// Explicit feature centering can still induce original-scale intercepts.
    pub fn fit_intercept(mut self, enabled: bool) -> Self {
        self.fit_intercept = enabled;
        self
    }

    /// Borrow response columns in response-major order.
    pub fn response(&self) -> &[f64] {
        self.response
    }

    /// Number of observations per response.
    pub fn nobs(&self) -> usize {
        self.nobs
    }

    /// Number of response columns.
    pub fn nresponses(&self) -> usize {
        self.nresponses
    }

    /// Whether fitting includes a free intercept for each response.
    pub fn fits_intercept(&self) -> bool {
        self.fit_intercept
    }

    pub(crate) fn validate<E>(&self, nrows: usize) -> Result<(), FitError<E>> {
        if self.nobs == 0 || self.nresponses == 0 {
            return Err(invalid(
                "multivariate Gaussian data must have observations and responses",
            ));
        }
        if self.nobs != nrows {
            return Err(invalid(
                "datafit observation count must match training rows",
            ));
        }
        let len = self
            .nobs
            .checked_mul(self.nresponses)
            .ok_or_else(|| invalid("response dimensions overflow usize"))?;
        if self.response.len() != len {
            return Err(invalid(
                "response length must match observation and response dimensions",
            ));
        }
        for (i, value) in self.response.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "response at observation {}, response {} is not finite",
                    i % self.nobs,
                    i / self.nobs
                )));
            }
        }
        Ok(())
    }

    fn validate_predictor(&self, predictor: &[f64]) -> Result<(), FitError> {
        self.validate(self.nobs)?;
        if predictor.len() != self.response.len() {
            return Err(invalid("response and predictor lengths must match"));
        }
        if predictor.iter().any(|value| !value.is_finite()) {
            return Err(invalid("predictor values must be finite"));
        }
        Ok(())
    }

    /// Write `(predictor - Y) / nobs` in response-major order.
    ///
    /// # Errors
    /// Reject invalid dimensions, nonfinite entries, or nonfinite arithmetic.
    /// Output is unspecified after an error.
    pub fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.validate_predictor(predictor)?;
        if output.len() != predictor.len() {
            return Err(invalid("predictor and gradient lengths must match"));
        }
        for ((&eta, &target), derivative) in predictor.iter().zip(self.response).zip(output) {
            *derivative = finite(
                (eta - target) / self.nobs as f64,
                "computing the multivariate predictor gradient",
                0,
            )?;
        }
        Ok(())
    }
}

impl Datafit for MultivariateGaussian<'_> {
    type Error = FitError;

    fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
        self.validate_predictor(predictor)?;
        Ok(Gaussian::residual_loss(
            self.response.iter().zip(predictor).map(|(y, eta)| y - eta),
            self.nobs,
            0,
        )?)
    }
}

// The shared vector iteration sees stacked observations. Its normalization
// remains the multivariate loss's 1/nobs, and intercepts enter through the design.
pub(crate) struct StackedGaussian<'a>(pub(crate) &'a MultivariateGaussian<'a>);

impl Datafit for StackedGaussian<'_> {
    type Error = FitError;
    fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
        self.0.value(predictor)
    }
}

impl SmoothDatafit for StackedGaussian<'_> {
    fn nobs(&self) -> usize {
        self.0.response.len()
    }
    fn fits_intercept(&self) -> bool {
        false
    }
    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.0.gradient(predictor, output)
    }
}
