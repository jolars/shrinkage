//! Gaussian loss on a linear predictor.

use crate::error::{NumericalFailure, finite, invalid};
use crate::{Datafit, DiagonalHessianDatafit, FitError, SmoothDatafit};

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

impl DiagonalHessianDatafit for Gaussian<'_> {
    fn hessian_diagonal(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.value(predictor)?;
        if output.len() != predictor.len() {
            return Err(invalid("predictor and Hessian lengths must match"));
        }
        output.fill(1.0 / predictor.len() as f64);
        Ok(())
    }
}

impl SmoothDatafit for Gaussian<'_> {
    fn gap_safe(&self) -> Option<&dyn crate::GapSafeDatafit> {
        Some(self)
    }

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

impl crate::GapSafeDatafit for Gaussian<'_> {
    fn dual_candidate(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.gradient(predictor, output)?;
        for u in output.iter_mut() {
            *u = -*u;
        }
        if self.fit_intercept {
            let mean = output.iter().map(|u| u / output.len() as f64).sum::<f64>();
            for u in output.iter_mut() {
                *u -= mean;
            }
            let last = output.len() - 1;
            output[last] = -output[..last].iter().sum::<f64>();
        }
        self.finalize_dual(output)
    }

    fn finalize_dual(&self, dual: &mut [f64]) -> Result<(), FitError> {
        self.validate(dual.len())?;
        if self.fit_intercept {
            crate::screening::intercept::balance(dual, None)?;
        }
        Ok(())
    }

    fn dual_objective(&self, dual: &[f64]) -> Result<f64, FitError> {
        self.validate(dual.len())?;
        if self.fit_intercept && dual.iter().sum::<f64>() != 0.0 {
            return Err(invalid(
                "Gaussian dual point violates the free-intercept equality",
            ));
        }
        let n = dual.len() as f64;
        let reference = if self.fit_intercept {
            self.response[0]
        } else {
            0.0
        };
        let mut value = 0.0;
        for (&y, &u) in self.response.iter().zip(dual) {
            value += finite(
                ((y - reference) - 0.5 * n * u) * u,
                "evaluating the Gaussian dual",
                0,
            )?;
        }
        Ok(finite(value, "summing the Gaussian dual", 0)?)
    }

    fn strong_concavity(&self) -> f64 {
        self.response.len() as f64
    }

    fn fenchel_gap(&self, predictor: &[f64], dual: &[f64]) -> Result<f64, FitError> {
        self.validate(predictor.len())?;
        if dual.len() != predictor.len() {
            return Err(invalid("dual dimensions must match Gaussian observations"));
        }
        let n = predictor.len() as f64;
        let mut gap = 0.0;
        for ((&y, &eta), &u) in self.response.iter().zip(predictor).zip(dual) {
            let difference = ((y - eta) / n.sqrt()) - n.sqrt() * u;
            gap += 0.5 * difference * difference;
        }
        Ok(finite(gap, "computing the Gaussian Fenchel gap", 0)?)
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
