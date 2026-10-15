//! Nonnegative scaling of complete penalty values and proximal maps.

use crate::error::{finite, invalid};
use crate::{FitError, Penalty, ProximalPenalty};

/// A complete penalty multiplied by a finite, nonnegative scalar.
///
/// `Scaled::new(scale, penalty)` evaluates `scale * penalty.value(coefficients)`.
/// Parameters are checked during evaluation and fitting. For a positive scale,
/// the exact proximal map delegates to the inner map with `step * scale`.
/// Zero scaling returns zero value and the identity proximal map, while still
/// checking the wrapped penalty through [`Penalty::validate_input`]. The
/// wrapper adds no allocations or coefficient storage.
///
/// ```
/// use shrinkage::{ElasticNet, Penalty, ProximalPenalty, Scaled};
///
/// let penalty = Scaled::new(0.5, ElasticNet::new(2.0, 4.0));
/// assert_eq!(penalty.value(&[-2.0])?, 6.0);
/// let mut output = [0.0];
/// penalty.prox(&[-4.0], 0.5, &mut output)?;
/// assert_eq!(output, [-1.75]);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
///
/// Scaling a value-only term does not supply a complete proximal map:
///
/// ```compile_fail
/// use shrinkage::{FitError, Penalty, ProximalPenalty, Scaled};
/// struct ValueOnly;
/// impl Penalty for ValueOnly {
///     type Error = FitError;
///     fn value(&self, _: &[f64]) -> Result<f64, FitError> { Ok(1.0) }
/// }
/// let penalty = Scaled::new(0.5, ValueOnly);
/// penalty.prox(&[1.0], 1.0, &mut [0.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Scaled<P> {
    scale: f64,
    penalty: P,
}

impl<P> Scaled<P> {
    /// Set the scalar and complete inner penalty, checked when used.
    pub fn new(scale: f64, penalty: P) -> Self {
        Self { scale, penalty }
    }

    /// Configured multiplier of the complete penalty.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Borrow the complete inner penalty, including its own strengths.
    pub fn penalty(&self) -> &P {
        &self.penalty
    }

    fn validate(&self, coefficients: &[f64]) -> Result<(), FitError> {
        if !self.scale.is_finite() || self.scale < 0.0 {
            return Err(invalid("penalty scale must be finite and nonnegative"));
        }
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(invalid("coefficients must be finite"));
        }
        Ok(())
    }
}

impl<P: Penalty<Error = FitError>> Penalty for Scaled<P> {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.validate(coefficients)?;
        if self.scale == 0.0 {
            self.penalty.validate_input(coefficients)?;
            return Ok(0.0);
        }
        Ok(finite(
            self.scale * self.penalty.value(coefficients)?,
            "scaling the penalty value",
            0,
        )?)
    }

    fn validate_input(&self, coefficients: &[f64]) -> Result<(), FitError> {
        self.validate(coefficients)?;
        self.penalty.validate_input(coefficients)
    }
}

impl<P: ProximalPenalty> ProximalPenalty for Scaled<P> {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.validate(input)?;
        if input.len() != output.len() {
            return Err(invalid("proximal input and output lengths must match"));
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid(
                "proximal step must be finite and strictly positive",
            ));
        }
        if self.scale == 0.0 {
            self.penalty.validate_input(input)?;
            output.copy_from_slice(input);
            return Ok(());
        }
        let scaled_step = finite(step * self.scale, "scaling a proximal step", 0)?;
        if scaled_step == 0.0 {
            return Err(FitError::NumericalFailure {
                operation: "scaling a proximal step",
                iteration: 0,
            });
        }
        self.penalty.prox(input, scaled_step, output)
    }
}
