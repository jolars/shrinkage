//! Sorted L1 evaluation and exact isotonic proximal minimization.

use std::cell::RefCell;

use crate::error::{finite, invalid};
use crate::{FitError, Penalty, ProximalPenalty};

/// SLOPE penalty `sum_j weights[j] * |theta|_(j)` on optimization coefficients.
///
/// `|theta|_(j)` denotes the jth largest coefficient magnitude. Supply one
/// finite, nonnegative, nonincreasing weight per coefficient, including the
/// penalty strength. Equal weights give L1; all zero weights give no penalty.
/// An empty weight vector defines the penalty on an empty coefficient vector.
/// Intercepts are excluded by the fitting APIs.
///
/// The exact proximal map sorts magnitudes, projects the thresholded values
/// onto nonincreasing, nonnegative sequences by pooling adjacent violations,
/// and restores the original signs and order. Sorting costs `O(p log p)` and
/// pooling costs `O(p)`. Owned CPU scratch storage is reused across calls.
/// Typed Gaussian problems also support [`crate::Hybrid`], combining full PG
/// transitions with exact signed magnitude-cluster sweeps.
///
/// ```
/// use shrinkage::{Penalty, ProximalPenalty, Slope};
///
/// let penalty = Slope::new(vec![3.0, 1.0]);
/// assert_eq!(penalty.value(&[-3.0, 3.0])?, 12.0);
/// let mut output = [0.0; 2];
/// penalty.prox(&[-3.0, 3.0], 1.0, &mut output)?;
/// assert_eq!(output, [-1.0, 1.0]);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Slope {
    weights: Vec<f64>,
    scratch: RefCell<Workspace>,
}

#[derive(Clone, Debug)]
struct Workspace {
    order: Vec<usize>,
    blocks: Vec<Block>,
}

#[derive(Clone, Copy, Debug)]
struct Block {
    start: usize,
    end: usize,
    mean: f64,
}

impl Slope {
    /// Set complete rank weights, checked during evaluation and fitting.
    pub fn new(weights: Vec<f64>) -> Self {
        let length = weights.len();
        Self {
            weights,
            scratch: RefCell::new(Workspace {
                order: (0..length).collect(),
                blocks: Vec::with_capacity(length),
            }),
        }
    }

    /// Configured rank weights, including the penalty strength.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn validate(&self, input: &[f64]) -> Result<(), FitError> {
        if self
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(invalid("SLOPE weights must be finite and nonnegative"));
        }
        if self.weights.windows(2).any(|pair| pair[0] < pair[1]) {
            return Err(invalid("SLOPE weights must be nonincreasing"));
        }
        if input.len() != self.weights.len() {
            return Err(invalid("SLOPE requires one weight per coefficient"));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(invalid("SLOPE input must be finite"));
        }
        Ok(())
    }
}

impl Workspace {
    fn sort(&mut self, input: &[f64]) {
        self.order
            .sort_unstable_by(|&a, &b| input[b].abs().total_cmp(&input[a].abs()).then(a.cmp(&b)));
    }
}

impl Penalty for Slope {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.validate(coefficients)?;
        let mut scratch = self.scratch.borrow_mut();
        scratch.sort(coefficients);
        let value = scratch
            .order
            .iter()
            .zip(&self.weights)
            .map(|(&index, &weight)| weight * coefficients[index].abs())
            .sum();
        Ok(finite(value, "computing the SLOPE penalty", 0)?)
    }

    fn validate_input(&self, coefficients: &[f64]) -> Result<(), FitError> {
        self.validate(coefficients)
    }
}

impl ProximalPenalty for Slope {
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
        let mut scratch = self.scratch.borrow_mut();
        scratch.sort(input);
        let Workspace { order, blocks } = &mut *scratch;
        blocks.clear();
        for (rank, (&index, &weight)) in order.iter().zip(&self.weights).enumerate() {
            let threshold = finite(step * weight, "computing a SLOPE proximal threshold", 0)?;
            blocks.push(Block {
                start: rank,
                end: rank + 1,
                mean: input[index].abs() - threshold,
            });
            while blocks.len() > 1 {
                let last = blocks.len() - 1;
                let (left, right) = (blocks[last - 1], blocks[last]);
                if left.mean > right.mean {
                    break;
                }
                // Weighted means avoid overflowing block sums when the
                // proximal minimizer itself is representable.
                let mean = if left.mean == right.mean {
                    left.mean
                } else {
                    let fraction =
                        (right.end - right.start) as f64 / (right.end - left.start) as f64;
                    left.mean * (1.0 - fraction) + right.mean * fraction
                };
                blocks[last - 1] = Block {
                    start: left.start,
                    end: right.end,
                    mean: finite(mean, "pooling SLOPE proximal blocks", 0)?,
                };
                blocks.pop();
            }
        }
        for block in blocks.iter() {
            for &index in &order[block.start..block.end] {
                output[index] = block.mean.max(0.0).copysign(input[index]);
            }
        }
        Ok(())
    }
}
