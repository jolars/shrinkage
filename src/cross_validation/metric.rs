use crate::error::{finite, invalid};
use crate::{Binomial, Datafit, FitError, Gaussian, NullModelDatafit, SmoothDatafit};

/// Validation losses averaged over held-out observations; lower is better.
///
/// Prediction inputs are original-scale linear predictors, including logits
/// for binomial scores. Scores exclude the penalty and use no training-size
/// correction. Binomial deviance subtracts the fractional-response saturated
/// loss before doubling, whereas log loss is the negative log likelihood.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationMetric {
    /// Gaussian mean squared error, twice the fitting loss.
    MeanSquaredError,
    /// Gaussian mean absolute prediction error.
    MeanAbsoluteError,
    /// Binomial mean negative log likelihood, evaluated stably from logits.
    LogLoss,
    /// Binomial mean deviance, twice the excess loss above saturation.
    BinomialDeviance,
}

impl ValidationMetric {
    /// Evaluate a score on equally weighted held-out observations.
    ///
    /// # Errors
    /// Reject empty or mismatched inputs, invalid responses, and nonfinite
    /// predictors or arithmetic. Binomial responses must lie in `[0, 1]`.
    pub fn score(self, response: &[f64], predictor: &[f64]) -> Result<f64, FitError> {
        let score = match self {
            Self::MeanSquaredError => 2.0 * Gaussian::new(response).value(predictor)?,
            Self::MeanAbsoluteError => {
                if response.is_empty()
                    || response.len() != predictor.len()
                    || response.iter().chain(predictor).any(|v| !v.is_finite())
                {
                    return Err(invalid(
                        "scores require nonempty, matching finite responses and predictors",
                    ));
                }
                response
                    .iter()
                    .zip(predictor)
                    .map(|(y, eta)| (y - eta).abs() / response.len() as f64)
                    .sum()
            }
            Self::LogLoss => Binomial::new(response).value(predictor)?,
            Self::BinomialDeviance => Binomial::new(response).deviance(predictor)?.unwrap(),
        };
        Ok(finite(score, "computing a validation score", 0)?)
    }
}

/// A univariate datafit that can be rebuilt on a training-fold response.
///
/// Implementations preserve the family and intercept setting. They validate
/// training responses before solver preparation and supply a compatible score.
pub trait CrossValidationDatafit: SmoothDatafit {
    /// Same family and settings borrowing a replacement response.
    type WithResponse<'a>: CrossValidationDatafit;
    /// Observed response in original row order.
    fn response(&self) -> &[f64];
    /// Rebuild the family without retaining observations from other folds.
    fn with_response<'a>(&self, response: &'a [f64]) -> Self::WithResponse<'a>;
    /// Default held-out metric for this family.
    fn validation_metric(&self) -> ValidationMetric;
    /// Whether this family supports the requested metric.
    fn supports_metric(&self, metric: ValidationMetric) -> bool;
    /// Validate a nonempty response, including degenerate training-fold policy.
    ///
    /// # Errors
    /// Reject invalid responses or training folds without a finite null model.
    fn validate_training(&self) -> Result<(), FitError>;
}

impl CrossValidationDatafit for Gaussian<'_> {
    type WithResponse<'a> = Gaussian<'a>;
    fn response(&self) -> &[f64] {
        self.response()
    }
    fn with_response<'a>(&self, response: &'a [f64]) -> Gaussian<'a> {
        Gaussian::new(response).fit_intercept(self.fits_intercept())
    }
    fn validation_metric(&self) -> ValidationMetric {
        ValidationMetric::MeanSquaredError
    }
    fn supports_metric(&self, metric: ValidationMetric) -> bool {
        matches!(
            metric,
            ValidationMetric::MeanSquaredError | ValidationMetric::MeanAbsoluteError
        )
    }
    fn validate_training(&self) -> Result<(), FitError> {
        self.validate(self.response().len())
    }
}

impl CrossValidationDatafit for Binomial<'_> {
    type WithResponse<'a> = Binomial<'a>;
    fn response(&self) -> &[f64] {
        self.response()
    }
    fn with_response<'a>(&self, response: &'a [f64]) -> Binomial<'a> {
        Binomial::new(response).fit_intercept(self.fits_intercept())
    }
    fn validation_metric(&self) -> ValidationMetric {
        ValidationMetric::LogLoss
    }
    fn supports_metric(&self, metric: ValidationMetric) -> bool {
        matches!(
            metric,
            ValidationMetric::LogLoss | ValidationMetric::BinomialDeviance
        )
    }
    fn validate_training(&self) -> Result<(), FitError> {
        // A boundary-only fold has no finite optimum for a free intercept,
        // even when a solver's numerical stopping check accepts a large logit.
        self.value(&vec![0.0; self.response().len()])?;
        if self.fits_intercept()
            && (!self.response().iter().any(|&y| y > 0.0)
                || !self.response().iter().any(|&y| y < 1.0))
        {
            return Err(invalid(
                "binomial training folds with a free intercept require positive success and failure totals; revise the folds or disable the intercept",
            ));
        }
        Ok(())
    }
}
