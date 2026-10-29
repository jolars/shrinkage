use crate::error::{finite, invalid};
use crate::{Binomial, Datafit, FitError, Gaussian, MultivariateGaussian, SmoothDatafit};

/// An exact zero-coefficient model with optimized, unpenalized intercepts.
///
/// This opt-in capability is independent of fitting arbitrary explicit paths.
pub trait NullModelDatafit {
    /// Return one finite optimization-scale intercept per response, or zero
    /// for each disabled intercept. Validate the response before returning.
    ///
    /// # Errors
    /// Reject invalid responses or a null model without a finite optimum.
    fn null_intercepts(&self) -> Result<Vec<f64>, FitError>;

    /// Evaluate twice the loss above its saturated value, averaged over rows.
    /// Return `None` when deviance is unavailable; automatic paths then fit
    /// their entire sequence. Support must be consistent across predictors.
    ///
    /// # Errors
    /// Reject invalid responses, predictors, or nonfinite arithmetic.
    fn deviance(&self, _predictor: &[f64]) -> Result<Option<f64>, FitError> {
        Ok(None)
    }
}

/// Object-safe smooth datafit that also supports automatic path initialization.
pub trait PathDatafit: SmoothDatafit + NullModelDatafit {}
impl<D: SmoothDatafit + NullModelDatafit + ?Sized> PathDatafit for D {}

impl NullModelDatafit for Gaussian<'_> {
    fn deviance(&self, predictor: &[f64]) -> Result<Option<f64>, FitError> {
        Ok(Some(finite(
            2.0 * self.value(predictor)?,
            "computing Gaussian deviance",
            0,
        )?))
    }

    fn null_intercepts(&self) -> Result<Vec<f64>, FitError> {
        self.validate(self.response().len())?;
        let mean = if self.fits_intercept() {
            finite(
                self.response()
                    .iter()
                    .map(|y| y / self.response().len() as f64)
                    .sum(),
                "computing the null intercept",
                0,
            )?
        } else {
            0.0
        };
        Ok(vec![mean])
    }
}

impl NullModelDatafit for Binomial<'_> {
    fn deviance(&self, predictor: &[f64]) -> Result<Option<f64>, FitError> {
        self.value(predictor)?;
        let n = self.response().len() as f64;
        let mut deviance = 0.0;
        for (&eta, &y) in predictor.iter().zip(self.response()) {
            let linear = if eta >= 0.0 {
                (1.0 - y) * eta
            } else {
                -y * eta
            };
            let entropy = if y == 0.0 || y == 1.0 {
                0.0
            } else {
                y * y.ln() + (1.0 - y) * (-y).ln_1p()
            };
            // Subtract saturation per row so fractional responses have zero
            // deviance at their own means. Clamp cancellation at saturation.
            let excess = (linear + (-eta.abs()).exp().ln_1p() + entropy).max(0.0);
            deviance += excess / n;
        }
        Ok(Some(finite(
            2.0 * deviance,
            "computing binomial deviance",
            0,
        )?))
    }

    fn null_intercepts(&self) -> Result<Vec<f64>, FitError> {
        self.value(&vec![0.0; self.response().len()])?;
        if !self.fits_intercept() {
            return Ok(vec![0.0]);
        }
        // The common observation count cancels in the odds. Summing before
        // division preserves subnormal positive totals and avoids rounding
        // the mean to a boundary when the finite null intercept still exists.
        let successes: f64 = self.response().iter().sum();
        let failures: f64 = self.response().iter().map(|y| 1.0 - y).sum();
        if successes <= 0.0 || failures <= 0.0 {
            return Err(invalid(
                "automatic binomial paths require a finite null intercept; disable the intercept or use an explicit path for constant boundary responses",
            ));
        }
        Ok(vec![finite(
            successes.ln() - failures.ln(),
            "computing the null intercept",
            0,
        )?])
    }
}

impl NullModelDatafit for MultivariateGaussian<'_> {
    fn deviance(&self, predictor: &[f64]) -> Result<Option<f64>, FitError> {
        Ok(Some(finite(
            2.0 * self.value(predictor)?,
            "computing multivariate Gaussian deviance",
            0,
        )?))
    }

    fn null_intercepts(&self) -> Result<Vec<f64>, FitError> {
        self.validate(self.nobs())?;
        self.response()
            .chunks_exact(self.nobs())
            .map(|response| {
                Ok(Gaussian::new(response)
                    .fit_intercept(self.fits_intercept())
                    .null_intercepts()?[0])
            })
            .collect()
    }
}

impl<D: Datafit + ?Sized> Datafit for Box<D> {
    type Error = D::Error;
    fn value(&self, predictor: &[f64]) -> Result<f64, Self::Error> {
        (**self).value(predictor)
    }
}
impl<D: SmoothDatafit + ?Sized> SmoothDatafit for Box<D> {
    fn gap_safe(&self) -> Option<&dyn crate::GapSafeDatafit> {
        (**self).gap_safe()
    }

    fn nobs(&self) -> usize {
        (**self).nobs()
    }
    fn fits_intercept(&self) -> bool {
        (**self).fits_intercept()
    }
    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        (**self).gradient(predictor, output)
    }
}
impl<D: NullModelDatafit + ?Sized> NullModelDatafit for Box<D> {
    fn null_intercepts(&self) -> Result<Vec<f64>, FitError> {
        (**self).null_intercepts()
    }
    fn deviance(&self, predictor: &[f64]) -> Result<Option<f64>, FitError> {
        (**self).deviance(predictor)
    }
}
