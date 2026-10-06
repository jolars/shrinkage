//! Penalty values and specialized coefficient subproblems.

use crate::FitError;
use crate::error::{finite, invalid};

/// An L1 penalty on optimization-scale coefficients.
#[derive(Clone, Copy, Debug)]
pub struct L1 {
    pub(crate) lambda: f64,
}

impl L1 {
    /// Set the penalty strength, checked when fitting.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

/// The value of a complete penalty on optimization-scale coefficients.
///
/// A value alone does not imply a proximal map or a valid coordinate update.
pub trait Penalty {
    /// Failure returned by penalty evaluation.
    type Error;

    /// Evaluate the penalty, including its strength.
    ///
    /// # Errors
    /// Returns an error for invalid coefficients, parameters, or arithmetic.
    fn value(&self, coefficients: &[f64]) -> Result<f64, Self::Error>;
}

impl L1 {
    /// Configured penalty strength.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub(crate) fn validate<E>(&self) -> Result<(), FitError<E>> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(invalid("lambda must be finite and nonnegative"));
        }
        Ok(())
    }

    pub(crate) fn value_unchecked(&self, coefficients: &[f64]) -> f64 {
        if self.lambda == 0.0 {
            0.0
        } else {
            coefficients
                .iter()
                .map(|value| self.lambda * value.abs())
                .sum()
        }
    }

    pub(crate) fn coordinate_minimum(&self, correlation: f64, curvature: f64) -> f64 {
        let thresholded = if correlation > self.lambda {
            correlation - self.lambda
        } else if correlation < -self.lambda {
            correlation + self.lambda
        } else {
            0.0
        };
        thresholded / curvature
    }
}

impl Penalty for L1 {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, Self::Error> {
        self.validate()?;
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(invalid("coefficients must be finite"));
        }
        Ok(finite(
            self.value_unchecked(coefficients),
            "computing the penalty",
            0,
        )?)
    }
}
