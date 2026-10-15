//! Exact penalty restrictions with explicit unpenalized coordinates.

use crate::error::{finite, invalid};
use crate::{ElasticNet, FitError, L1, Penalty, ProximalPenalty, Ridge};

/// A convex penalty with an exact proximal map after excluding coordinates.
///
/// Each `true` entry in `unpenalized` excludes the corresponding coefficient.
/// The restricted value equals the complete penalty evaluated with those
/// coordinates set to zero. Its proximal map minimizes
/// `||z - input||² / (2 * step) + self.value_restricted(z, unpenalized)` and
/// copies excluded inputs unchanged. Masks have one entry per coefficient.
///
/// A complete proximal map alone does not imply this capability. In particular,
/// restoring excluded inputs after a coupled proximal map is generally wrong.
/// External penalties must implement the restricted problem explicitly.
pub trait RestrictedProximalPenalty: ProximalPenalty {
    /// Evaluate the penalty with excluded coordinates set to zero.
    ///
    /// # Errors
    /// Reject invalid parameters, mismatched mask lengths, nonfinite
    /// coefficients (including excluded ones), or nonfinite arithmetic.
    fn value_restricted(&self, coefficients: &[f64], unpenalized: &[bool])
    -> Result<f64, FitError>;

    /// Validate coefficients and a restriction without evaluating its value.
    ///
    /// The default checks mask length and calls [`Penalty::validate_input`]
    /// on the complete coefficients. Override it if restriction changes input
    /// compatibility or if complete validation performs unused value arithmetic.
    /// This check must still reject invalid parameters and nonfinite excluded
    /// coefficients, including when an outer [`super::Scaled`] has zero scale.
    ///
    /// # Errors
    /// Reject invalid parameters, coefficients, or mask lengths.
    fn validate_restricted_input(
        &self,
        coefficients: &[f64],
        unpenalized: &[bool],
    ) -> Result<(), FitError> {
        if coefficients.len() != unpenalized.len() {
            return Err(invalid(
                "unpenalized mask length must match the coefficients",
            ));
        }
        self.validate_input(coefficients)
    }

    /// Write the exact restricted proximal minimizer into reusable storage.
    ///
    /// # Errors
    /// Reject invalid parameters, nonpositive or nonfinite steps, mismatched
    /// input, output, or mask lengths, nonfinite input (including excluded
    /// coordinates), or nonfinite arithmetic. Output is unspecified on error.
    fn prox_restricted(
        &self,
        input: &[f64],
        unpenalized: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError>;
}

/// A complete penalty with explicit unpenalized coordinates.
///
/// `true` excludes a coordinate, and `false` retains its penalty. The mask must
/// have one entry per optimization-scale coefficient, excluding any intercept.
/// Parameters and mask length are checked during evaluation and fitting.
/// Excluded coordinates contribute no penalty and are copied unchanged by the
/// proximal map. The wrapper owns its mask and allocates no evaluation workspace.
///
/// ```
/// use shrinkage::{L1, Masked, Penalty, ProximalPenalty};
///
/// let penalty = Masked::new(vec![true, false], L1::new(2.0));
/// assert_eq!(penalty.value(&[10.0, -3.0])?, 6.0);
/// let mut output = [0.0; 2];
/// penalty.prox(&[10.0, -3.0], 0.5, &mut output)?;
/// assert_eq!(output, [10.0, -2.0]);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
///
/// A coupled penalty without an implemented restriction cannot be masked,
/// even if it supplies a complete proximal map:
///
/// ```compile_fail
/// use shrinkage::{Masked, Slope};
/// let penalty = Masked::new(vec![true, false], Slope::new(vec![2.0, 1.0]));
/// ```
///
/// Runtime selection must preserve the restricted capability before masking:
///
/// ```compile_fail
/// use shrinkage::{L1, Masked, ProximalPenalty};
/// let penalty: Box<dyn ProximalPenalty> = Box::new(L1::new(1.0));
/// let masked = Masked::new(vec![true, false], penalty);
/// ```
#[derive(Clone, Debug)]
pub struct Masked<P> {
    unpenalized: Vec<bool>,
    penalty: P,
}

impl<P: RestrictedProximalPenalty> Masked<P> {
    /// Set the unpenalized mask and inner penalty, checked when used.
    pub fn new(unpenalized: Vec<bool>, penalty: P) -> Self {
        Self {
            unpenalized,
            penalty,
        }
    }

    /// One flag per coefficient; `true` marks an unpenalized coordinate.
    pub fn unpenalized(&self) -> &[bool] {
        &self.unpenalized
    }

    /// Borrow the complete inner penalty, including its strengths.
    pub fn penalty(&self) -> &P {
        &self.penalty
    }
}

impl<P: RestrictedProximalPenalty> Penalty for Masked<P> {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.penalty
            .value_restricted(coefficients, &self.unpenalized)
    }

    fn validate_input(&self, coefficients: &[f64]) -> Result<(), FitError> {
        self.penalty
            .validate_restricted_input(coefficients, &self.unpenalized)
    }
}

impl<P: RestrictedProximalPenalty> ProximalPenalty for Masked<P> {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.penalty
            .prox_restricted(input, &self.unpenalized, step, output)
    }
}

impl RestrictedProximalPenalty for L1 {
    fn value_restricted(
        &self,
        coefficients: &[f64],
        unpenalized: &[bool],
    ) -> Result<f64, FitError> {
        self.validate_restricted_input(coefficients, unpenalized)?;
        let value = if self.lambda() == 0.0 {
            0.0
        } else {
            coefficients
                .iter()
                .zip(unpenalized)
                .filter(|(_, excluded)| !**excluded)
                .map(|(value, _)| self.lambda() * value.abs())
                .sum()
        };
        Ok(finite(value, "computing the restricted penalty", 0)?)
    }

    fn prox_restricted(
        &self,
        input: &[f64],
        unpenalized: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        ElasticNet::new(self.lambda(), 0.0).prox_restricted(input, unpenalized, step, output)
    }
}

impl RestrictedProximalPenalty for Ridge {
    fn value_restricted(
        &self,
        coefficients: &[f64],
        unpenalized: &[bool],
    ) -> Result<f64, FitError> {
        ElasticNet::new(0.0, self.lambda()).value_restricted(coefficients, unpenalized)
    }

    fn prox_restricted(
        &self,
        input: &[f64],
        unpenalized: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        ElasticNet::new(0.0, self.lambda()).prox_restricted(input, unpenalized, step, output)
    }
}

impl RestrictedProximalPenalty for ElasticNet {
    fn value_restricted(
        &self,
        coefficients: &[f64],
        unpenalized: &[bool],
    ) -> Result<f64, FitError> {
        self.validate_restricted_input(coefficients, unpenalized)?;
        let l1 = L1::new(self.l1()).value_restricted(coefficients, unpenalized)?;
        let l2 = if self.l2() == 0.0 {
            0.0
        } else {
            coefficients
                .iter()
                .zip(unpenalized)
                .filter(|(_, excluded)| !**excluded)
                .map(|(&coefficient, _)| {
                    let scaled = self.l2().sqrt() * coefficient;
                    0.5 * scaled * scaled
                })
                .sum()
        };
        Ok(finite(l1 + l2, "computing the restricted penalty", 0)?)
    }

    fn prox_restricted(
        &self,
        input: &[f64],
        unpenalized: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        self.validate_restricted_input(input, unpenalized)?;
        if input.len() != output.len() {
            return Err(invalid("proximal input and output lengths must match"));
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid(
                "proximal step must be finite and strictly positive",
            ));
        }
        if unpenalized.iter().all(|&excluded| excluded) {
            output.copy_from_slice(input);
            return Ok(());
        }
        let threshold = finite(step * self.l1(), "computing a proximal threshold", 0)?;
        let denominator = finite(1.0 + step * self.l2(), "computing proximal curvature", 0)?;
        for ((&value, &excluded), target) in input.iter().zip(unpenalized).zip(output) {
            *target = if excluded {
                value
            } else {
                value.signum() * (value.abs() - threshold).max(0.0) / denominator
            };
        }
        Ok(())
    }
}

impl<P: RestrictedProximalPenalty + ?Sized> RestrictedProximalPenalty for Box<P> {
    fn value_restricted(
        &self,
        coefficients: &[f64],
        unpenalized: &[bool],
    ) -> Result<f64, FitError> {
        (**self).value_restricted(coefficients, unpenalized)
    }

    fn validate_restricted_input(
        &self,
        coefficients: &[f64],
        unpenalized: &[bool],
    ) -> Result<(), FitError> {
        (**self).validate_restricted_input(coefficients, unpenalized)
    }

    fn prox_restricted(
        &self,
        input: &[f64],
        unpenalized: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        (**self).prox_restricted(input, unpenalized, step, output)
    }
}
