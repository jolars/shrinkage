//! Weighted adjacent fusion and its complete taut-string proximal map.

use std::{cell::RefCell, collections::VecDeque};

use crate::error::{finite, invalid};
use crate::{FitError, Penalty, ProximalPenalty};

/// Complete penalty `l1 * sum_j |beta[j]| + fusion * sum_j w[j] * |beta[j+1]-beta[j]|`.
///
/// Edges follow feature order and contribute once. Both strengths and every
/// weight must be finite and nonnegative, checked on use. Unit weights are
/// implicit unless [`Self::with_weights`] supplies exactly one weight per edge.
/// Zero weights cut the chain. Empty and single-feature chains have no edges.
/// Intercepts are excluded, and coefficients refer to the optimization scale.
///
/// The exact complete prox uses weighted taut strings followed by uniform L1
/// soft-thresholding. Value and prox take `O(p)` time; the prox reuses `O(p)`
/// CPU storage. [`crate::ProximalGradient`] supports typed and runtime problems,
/// and [`crate::Hybrid`] adds exact Gaussian segment-coordinate sweeps.
///
/// ```
/// use shrinkage::{ChainFusedLasso, Penalty, ProximalPenalty};
/// let penalty = ChainFusedLasso::new(0.1, 0.2).with_weights(vec![2.0, 0.0]);
/// assert!((penalty.value(&[-2.0, 2.0, 1.0])? - 2.1).abs() < 1e-14);
/// let mut output = [0.0; 3];
/// penalty.prox(&[-2.0, 2.0, 1.0], 1.0, &mut output)?;
/// assert_eq!(output, [-1.5, 1.5, 0.9]);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
#[derive(Clone, Debug)]
pub struct ChainFusedLasso {
    l1: f64,
    fusion: f64,
    weights: Option<Vec<f64>>,
    scratch: RefCell<Workspace>,
}

// Compensation preserves small signal increments when tube widths are large.
#[derive(Clone, Copy, Debug, Default)]
struct Height {
    sum: f64,
    correction: f64,
}

impl Height {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn combine(&mut self, other: Self, sign: f64) {
        self.add(sign * other.sum);
        self.add(sign * other.correction);
    }

    fn total(self) -> Result<f64, FitError> {
        Ok(finite(
            self.sum + self.correction,
            "forming a chain taut string",
            0,
        )?)
    }
}

#[derive(Clone, Copy, Debug)]
struct Piece {
    length: usize,
    rise: Height,
    slope: f64,
}

impl Piece {
    fn new(length: usize, rise: Height) -> Result<Self, FitError> {
        Ok(Self {
            length,
            rise,
            slope: rise.total()? / length as f64,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct Workspace {
    upper: VecDeque<Piece>,
    lower: VecDeque<Piece>,
    widths: Vec<f64>,
    shifted: Vec<f64>,
    negative: Vec<f64>,
    screening_lower: Vec<f64>,
    screening_upper: Vec<f64>,
}

impl ChainFusedLasso {
    /// Set independent L1 and fusion strengths, with implicit unit weights.
    pub fn new(l1: f64, fusion: f64) -> Self {
        Self {
            l1,
            fusion,
            weights: None,
            scratch: RefCell::new(Workspace::default()),
        }
    }

    /// Supply one finite, nonnegative weight per adjacent edge, checked on use.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Configured L1 strength.
    pub fn l1(&self) -> f64 {
        self.l1
    }

    /// Configured multiplier of the edge weights.
    pub fn fusion(&self) -> f64 {
        self.fusion
    }

    /// Explicit edge weights, or `None` for implicit unit weights.
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    pub(crate) fn weight(&self, edge: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |weights| weights[edge])
    }

    pub(crate) fn with_screening_bounds<T>(
        &self,
        length: usize,
        inspect: impl FnOnce(&mut [f64], &mut [f64]) -> T,
    ) -> T {
        let mut scratch = self.scratch.borrow_mut();
        scratch.screening_lower.resize(length, 0.0);
        scratch.screening_upper.resize(length, 0.0);
        let Workspace {
            screening_lower,
            screening_upper,
            ..
        } = &mut *scratch;
        inspect(screening_lower, screening_upper)
    }
}

impl Penalty for ChainFusedLasso {
    type Error = FitError;

    fn validate_input(&self, coefficients: &[f64]) -> Result<(), FitError> {
        if !self.l1.is_finite() || self.l1 < 0.0 || !self.fusion.is_finite() || self.fusion < 0.0 {
            return Err(invalid(
                "chain fused-lasso strengths must be finite and nonnegative",
            ));
        }
        if let Some(weights) = &self.weights {
            if weights.len() != coefficients.len().saturating_sub(1) {
                return Err(invalid(
                    "chain edge weights must match the coefficient count minus one",
                ));
            }
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err(invalid("chain edge weights must be finite and nonnegative"));
            }
        }
        if coefficients.iter().any(|x| !x.is_finite()) {
            return Err(invalid("chain fused-lasso coefficients must be finite"));
        }
        Ok(())
    }

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.validate_input(coefficients)?;
        let mut value = if self.l1 == 0.0 {
            0.0
        } else {
            coefficients.iter().map(|x| self.l1 * x.abs()).sum()
        };
        if self.fusion != 0.0 {
            for (edge, pair) in coefficients.windows(2).enumerate() {
                let weight = self.weight(edge);
                if weight != 0.0 && pair[0] != pair[1] {
                    value += (self.fusion * weight) * (pair[1] - pair[0]).abs();
                }
            }
        }
        Ok(finite(value, "computing the chain fused-lasso penalty", 0)?)
    }
}

fn threshold(strength: f64, step: f64) -> Result<f64, FitError> {
    let value = finite(strength * step, "scaling a chain proximal threshold", 0)?;
    if strength > 0.0 && value == 0.0 {
        return Err(FitError::NumericalFailure {
            operation: "scaling chain proximal thresholds without underflow",
            iteration: 0,
        });
    }
    Ok(value)
}

impl ProximalPenalty for ChainFusedLasso {
    fn gap_safe(&self) -> Option<&dyn crate::screening::GapSafePenalty> {
        Some(self)
    }

    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.validate_input(input)?;
        if input.len() != output.len() {
            return Err(invalid("proximal input and output lengths must match"));
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid(
                "proximal step must be finite and strictly positive",
            ));
        }
        if input.is_empty() {
            return Ok(());
        }
        let l1 = threshold(self.l1, step)?;
        let mut scratch = self.scratch.borrow_mut();
        scratch.widths.clear();
        for edge in 0..input.len() - 1 {
            let weight = self.weight(edge);
            let width = if self.fusion == 0.0 || weight == 0.0 {
                0.0
            } else {
                threshold(threshold(self.fusion, weight)?, step)?
            };
            scratch.widths.push(width);
        }
        let Workspace {
            upper,
            lower,
            widths,
            ..
        } = &mut *scratch;
        let mut start = 0;
        for end in 1..=input.len() {
            if end == input.len() || widths[end - 1] == 0.0 {
                taut_string(
                    &input[start..end],
                    &widths[start..end - 1],
                    &mut output[start..end],
                    upper,
                    lower,
                )?;
                start = end;
            }
        }
        // Uniform soft-thresholding preserves every nonzero edge difference's
        // sign, so the TV dual remains a subgradient of the thresholded point.
        for value in output {
            *value = (value.abs() - l1).max(0.0).copysign(*value);
            if *value == 0.0 {
                *value = 0.0;
            }
        }
        Ok(())
    }
}

impl ChainFusedLasso {
    pub(crate) fn prox_screened(
        &self,
        input: &[f64],
        zeroed: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        self.validate_input(input)?;
        if input.len() != output.len() || input.len() != zeroed.len() {
            return Err(invalid(
                "zero restriction dimensions must match the coefficients",
            ));
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid(
                "proximal step must be finite and strictly positive",
            ));
        }
        if !zeroed.iter().any(|&zero| zero) {
            return self.prox(input, step, output);
        }
        output.fill(0.0);
        if zeroed.iter().all(|&zero| zero) {
            return Ok(());
        }
        let l1 = threshold(self.l1, step)?;
        let mut scratch = self.scratch.borrow_mut();
        scratch.widths.clear();
        for edge in 0..input.len().saturating_sub(1) {
            let weight = self.weight(edge);
            let width = if self.fusion == 0.0 || weight == 0.0 {
                0.0
            } else {
                threshold(threshold(self.fusion, weight)?, step)?
            };
            scratch.widths.push(width);
        }
        scratch.shifted.resize(input.len(), 0.0);
        scratch.negative.resize(input.len(), 0.0);
        let Workspace {
            upper,
            lower,
            widths,
            shifted,
            negative,
            ..
        } = &mut *scratch;
        let mut start = 0;
        while start < input.len() {
            if zeroed[start] {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < input.len() && !zeroed[end] && widths[end - 1] != 0.0 {
                end += 1;
            }
            for j in start..end {
                let unary = restricted_threshold(l1, widths, zeroed, j)?;
                shifted[j] = finite(
                    input[j] - unary,
                    "shifting a restricted chain proximal input",
                    0,
                )?;
            }
            taut_string(
                &shifted[start..end],
                &widths[start..end - 1],
                &mut output[start..end],
                upper,
                lower,
            )?;
            for j in start..end {
                let unary = restricted_threshold(l1, widths, zeroed, j)?;
                shifted[j] = finite(
                    input[j] + unary,
                    "shifting a restricted chain proximal input",
                    0,
                )?;
            }
            taut_string(
                &shifted[start..end],
                &widths[start..end - 1],
                &mut negative[start..end],
                upper,
                lower,
            )?;
            // TV separates across positive and negative parts. Its proximal
            // comparison principle makes these two clipped parts disjoint.
            for j in start..end {
                output[j] = output[j].max(0.0) + negative[j].min(0.0);
            }
            start = end;
        }
        Ok(())
    }
}

fn restricted_threshold(
    l1: f64,
    widths: &[f64],
    zeroed: &[bool],
    index: usize,
) -> Result<f64, FitError> {
    let left = if index > 0 && zeroed[index - 1] {
        widths[index - 1]
    } else {
        0.0
    };
    let right = if index + 1 < zeroed.len() && zeroed[index + 1] {
        widths[index]
    } else {
        0.0
    };
    finite(l1 + left + right, "forming a restricted chain threshold", 0).map_err(Into::into)
}

fn append(queue: &mut VecDeque<Piece>, mut piece: Piece, upper: bool) -> Result<(), FitError> {
    while let Some(last) = queue.back() {
        if if upper {
            last.slope <= piece.slope
        } else {
            last.slope >= piece.slope
        } {
            break;
        }
        let last = queue.pop_back().unwrap();
        piece.length += last.length;
        piece.rise.combine(last.rise, 1.0);
        piece.slope = piece.rise.total()? / piece.length as f64;
    }
    queue.push_back(piece);
    Ok(())
}

// Each hull piece enters and leaves a deque once. Unlike the linearized
// taut-string variant, retaining both hulls guarantees linear running time.
fn taut_string(
    input: &[f64],
    widths: &[f64],
    output: &mut [f64],
    upper: &mut VecDeque<Piece>,
    lower: &mut VecDeque<Piece>,
) -> Result<(), FitError> {
    upper.clear();
    lower.clear();
    let mut upper_height = Height::default();
    let mut lower_height = Height::default();
    let mut previous = 0.0;
    let mut start = 0;
    for (index, &value) in input.iter().enumerate() {
        let end = index + 1;
        let width = widths.get(index).copied().unwrap_or(0.0);
        let mut up = Height::default();
        up.add(value);
        up.add(width);
        up.add(-previous);
        let mut down = Height::default();
        down.add(value);
        down.add(-width);
        down.add(previous);
        upper_height.combine(up, 1.0);
        lower_height.combine(down, 1.0);
        append(upper, Piece::new(1, up)?, true)?;
        append(lower, Piece::new(1, down)?, false)?;
        while lower[0].slope > upper[0].slope && lower[0].length != upper[0].length {
            let take_upper = upper[0].length < lower[0].length;
            let piece = if take_upper {
                upper.pop_front()
            } else {
                lower.pop_front()
            }
            .unwrap();
            output[start..start + piece.length].fill(piece.slope);
            start += piece.length;
            upper_height.combine(piece.rise, -1.0);
            lower_height.combine(piece.rise, -1.0);
            let (opposite, height) = if take_upper {
                (&mut *lower, lower_height)
            } else {
                (&mut *upper, upper_height)
            };
            opposite.clear();
            opposite.push_back(Piece::new(end - start, height)?);
        }
        previous = width;
    }
    for piece in upper.iter() {
        output[start..start + piece.length].fill(piece.slope);
        start += piece.length;
    }
    Ok(())
}
