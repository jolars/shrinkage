//! Weighted disjoint L2 norms and their exact block proximal map.

use crate::error::{finite, invalid};
use crate::{DisjointGroups, FitError, Penalty, ProximalPenalty};

/// Group LASSO `lambda * sum_g weights[g] * ||theta[g]||₂`.
///
/// Groups address optimization-scale coefficients and exclude intercepts. The
/// supplied [`DisjointGroups`] fixes the coefficient count, explicit weights,
/// and coverage policy. There is no implicit group-size scaling. Zero-weight
/// groups and uncovered coefficients are unpenalized. Singleton groups give
/// weighted L1, and zero lambda gives the zero penalty.
///
/// The exact proximal map shrinks each block by
/// `max(1 - step * lambda * weight / ||input[g]||₂, 0)` and copies free
/// coordinates. Value evaluation and the prox take linear time and allocate no
/// scratch storage. Gaussian fits use [`crate::ProximalGradient`]'s reusable
/// product and iteration workspaces through both typed and runtime problems.
///
/// ```
/// use shrinkage::{DisjointGroups, GroupLasso, Penalty, ProximalPenalty, UncoveredPolicy};
///
/// let groups = DisjointGroups::new(
///     3, vec![vec![0, 2]], vec![2.0], UncoveredPolicy::Unpenalized,
/// )?;
/// let penalty = GroupLasso::new(0.5, groups);
/// assert_eq!(penalty.value(&[3.0, 9.0, 4.0])?, 5.0);
/// let mut output = [0.0; 3];
/// penalty.prox(&[3.0, 9.0, 4.0], 2.0, &mut output)?;
/// assert!((output[0] - 1.8).abs() < 1e-14);
/// assert_eq!(output[1], 9.0);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
#[derive(Clone, Debug)]
pub struct GroupLasso {
    lambda: f64,
    groups: DisjointGroups,
}

impl GroupLasso {
    /// Set the overall strength and validated groups. Strength is checked on use.
    pub fn new(lambda: f64, groups: DisjointGroups) -> Self {
        Self { lambda, groups }
    }

    /// Configured multiplier of all explicit group weights.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Immutable group definitions, weights, and coverage policy.
    pub fn groups(&self) -> &DisjointGroups {
        &self.groups
    }
}

// Keeping the scale separate avoids squaring large or tiny coefficients and
// lets the prox handle even a block whose norm exceeds f64::MAX.
fn scaled_norm(input: &[f64], group: &[usize]) -> (f64, f64) {
    let scale = group.iter().map(|&j| input[j].abs()).fold(0.0, f64::max);
    let unit_norm = if scale == 0.0 {
        0.0
    } else {
        group
            .iter()
            .map(|&j| (input[j] / scale).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    (scale, unit_norm)
}

impl Penalty for GroupLasso {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.validate_input(coefficients)?;
        let mut value = 0.0;
        if self.lambda != 0.0 {
            for (group, &weight) in self.groups.groups().iter().zip(self.groups.weights()) {
                if weight == 0.0 {
                    continue;
                }
                let (scale, unit_norm) = scaled_norm(coefficients, group);
                if scale != 0.0 {
                    let strength = finite(self.lambda * weight, "computing a group strength", 0)?;
                    value += (strength * scale) * unit_norm;
                }
            }
        }
        Ok(finite(value, "computing the Group LASSO penalty", 0)?)
    }

    fn validate_input(&self, coefficients: &[f64]) -> Result<(), FitError> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(invalid("lambda must be finite and nonnegative"));
        }
        if coefficients.len() != self.groups.ncoefficients() {
            return Err(invalid(
                "Group LASSO coefficient count must match the group definitions",
            ));
        }
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(invalid("Group LASSO coefficients must be finite"));
        }
        Ok(())
    }
}

impl ProximalPenalty for GroupLasso {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.validate_input(input)?;
        if output.len() != input.len() {
            return Err(invalid("proximal input and output lengths must match"));
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid(
                "proximal step must be finite and strictly positive",
            ));
        }
        output.copy_from_slice(input);
        if self.lambda == 0.0 {
            return Ok(());
        }
        for (group, &weight) in self.groups.groups().iter().zip(self.groups.weights()) {
            if weight == 0.0 {
                continue;
            }
            let strength = finite(self.lambda * weight, "computing a group strength", 0)?;
            let threshold = finite(step * strength, "computing a group proximal threshold", 0)?;
            let (scale, unit_norm) = scaled_norm(input, group);
            let factor = if scale == 0.0 {
                0.0
            } else {
                (1.0 - (threshold / scale) / unit_norm).max(0.0)
            };
            for &j in group {
                output[j] = factor * input[j];
            }
        }
        Ok(())
    }
}
