//! Apply complete vector penalties along explicit coefficient axes.

use std::cell::RefCell;

use crate::error::{finite, invalid};
use crate::{CoefficientLayout, FitError, Penalty, ProximalPenalty};

/// The coefficient vectors to which a complete penalty applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PenaltyAxis {
    /// Apply once to all coefficients in response-major order.
    All,
    /// Apply to the feature vector of each response, summing their values.
    Features,
    /// Apply to the response vector of each feature, summing their values.
    Responses,
}

/// A complete vector penalty applied along an explicit coefficient axis.
///
/// `Features` separates responses; `Responses` can couple their coefficients
/// at each feature. `All` can couple the full matrix. Each group uses the
/// supplied strength without dividing by the group or response count. The prox
/// is exact because the groups are disjoint. Intercepts are never included.
/// Strided response groups use owned reusable CPU scratch buffers.
///
/// Compose [`crate::Scaled`] and [`crate::Masked`] inside the axis penalty.
/// With `Responses`, each feature row receives the same vector penalty and
/// a mask has one entry per response. With `Features`, a mask has one entry
/// per feature and excludes those feature rows across all responses. Masks
/// require an inner [`crate::RestrictedProximalPenalty`].
///
/// ```
/// use shrinkage::{
///     AxisPenalty, CoefficientLayout, Masked, Penalty, PenaltyAxis,
///     ProximalPenalty, Ridge, Scaled,
/// };
///
/// let layout = CoefficientLayout::new(3, 2)?;
/// let penalty = AxisPenalty::new(
///     layout,
///     PenaltyAxis::Responses,
///     Scaled::new(0.5, Masked::new(vec![true, false], Ridge::new(2.0))),
/// );
/// // Each response's feature vector is contiguous, so the rows are [1, 4],
/// // [2, 5], and [3, 6]. Only the second response is penalized in each row.
/// let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// assert!((penalty.value(&input)? - 38.5).abs() < 1e-12);
/// let mut output = [0.0; 6];
/// penalty.prox(&input, 1.0, &mut output)?;
/// assert_eq!(output, [1.0, 2.0, 3.0, 2.0, 2.5, 3.0]);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
#[derive(Clone, Debug)]
pub struct AxisPenalty<P> {
    layout: CoefficientLayout,
    axis: PenaltyAxis,
    penalty: P,
    scratch: RefCell<(Vec<f64>, Vec<f64>)>,
}

impl<P> AxisPenalty<P> {
    /// Apply a vector penalty to the selected axis of the given layout.
    pub fn new(layout: CoefficientLayout, axis: PenaltyAxis, penalty: P) -> Self {
        let width = if axis == PenaltyAxis::Responses {
            layout.nresponses()
        } else {
            0
        };
        Self {
            layout,
            axis,
            penalty,
            scratch: RefCell::new((vec![0.0; width], vec![0.0; width])),
        }
    }

    /// Feature and response dimensions on which this penalty is defined.
    pub fn layout(&self) -> CoefficientLayout {
        self.layout
    }

    /// Axis along which the complete vector penalty applies.
    pub fn axis(&self) -> PenaltyAxis {
        self.axis
    }

    /// Borrow the complete vector penalty applied to each group.
    pub fn penalty(&self) -> &P {
        &self.penalty
    }

    fn validate(&self, input: &[f64]) -> Result<(), FitError> {
        if input.len() != self.layout.len() {
            return Err(invalid("coefficient length must match the penalty layout"));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(invalid("coefficients must be finite"));
        }
        Ok(())
    }
}

impl<P: Penalty<Error = FitError>> Penalty for AxisPenalty<P> {
    type Error = FitError;

    fn validate_input(&self, coefficients: &[f64]) -> Result<(), FitError> {
        self.validate(coefficients)?;
        if self.layout.is_empty() && self.axis == PenaltyAxis::Responses {
            return self.penalty.validate_input(&self.scratch.borrow().0);
        }
        if self.axis == PenaltyAxis::All || self.layout.is_empty() {
            return self.penalty.validate_input(coefficients);
        }
        match self.axis {
            PenaltyAxis::All => unreachable!(),
            PenaltyAxis::Features => {
                for group in coefficients.chunks_exact(self.layout.nfeatures()) {
                    self.penalty.validate_input(group)?;
                }
            }
            PenaltyAxis::Responses => {
                let mut scratch = self.scratch.borrow_mut();
                for feature in 0..self.layout.nfeatures() {
                    for (response, entry) in scratch.0.iter_mut().enumerate() {
                        *entry = coefficients[response * self.layout.nfeatures() + feature];
                    }
                    self.penalty.validate_input(&scratch.0)?;
                }
            }
        }
        Ok(())
    }

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.validate(coefficients)?;
        if self.layout.is_empty() {
            if self.axis == PenaltyAxis::Responses {
                // There are no response groups, but the inner penalty still
                // needs validation against its configured response width.
                self.penalty.value(&self.scratch.borrow().0)?;
                return Ok(0.0);
            }
            let value = self.penalty.value(coefficients)?;
            return Ok(finite(
                match self.axis {
                    PenaltyAxis::All => value,
                    PenaltyAxis::Features => value * self.layout.nresponses() as f64,
                    PenaltyAxis::Responses => 0.0,
                },
                "computing the matrix penalty",
                0,
            )?);
        }
        if self.axis == PenaltyAxis::All {
            return self.penalty.value(coefficients);
        }
        let mut value = 0.0;
        match self.axis {
            PenaltyAxis::All => unreachable!(),
            PenaltyAxis::Features => {
                for group in coefficients.chunks_exact(self.layout.nfeatures()) {
                    value += self.penalty.value(group)?;
                }
            }
            PenaltyAxis::Responses => {
                let mut scratch = self.scratch.borrow_mut();
                for feature in 0..self.layout.nfeatures() {
                    for (response, entry) in scratch.0.iter_mut().enumerate() {
                        *entry = coefficients[response * self.layout.nfeatures() + feature];
                    }
                    value += self.penalty.value(&scratch.0)?;
                }
            }
        }
        Ok(finite(value, "computing the matrix penalty", 0)?)
    }
}

impl<P: ProximalPenalty> ProximalPenalty for AxisPenalty<P> {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.validate(input)?;
        if output.len() != input.len() {
            return Err(invalid("proximal input and output lengths must match"));
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid(
                "proximal step must be finite and strictly positive",
            ));
        }
        if self.layout.is_empty() && self.axis == PenaltyAxis::Responses {
            let mut scratch = self.scratch.borrow_mut();
            let (group, target) = &mut *scratch;
            return self.penalty.prox(group, step, target);
        }
        if self.axis == PenaltyAxis::All || self.layout.is_empty() {
            return self.penalty.prox(input, step, output);
        }
        match self.axis {
            PenaltyAxis::All => unreachable!(),
            PenaltyAxis::Features => {
                for (group, target) in input
                    .chunks_exact(self.layout.nfeatures())
                    .zip(output.chunks_exact_mut(self.layout.nfeatures()))
                {
                    self.penalty.prox(group, step, target)?;
                }
            }
            PenaltyAxis::Responses => {
                let mut scratch = self.scratch.borrow_mut();
                let (group, target) = &mut *scratch;
                for feature in 0..self.layout.nfeatures() {
                    for (response, entry) in group.iter_mut().enumerate() {
                        *entry = input[response * self.layout.nfeatures() + feature];
                    }
                    self.penalty.prox(group, step, target)?;
                    for (response, &entry) in target.iter().enumerate() {
                        output[response * self.layout.nfeatures() + feature] = entry;
                    }
                }
            }
        }
        Ok(())
    }
}
