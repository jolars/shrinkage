//! Uniform all-pairs signed fusion and its complete isotonic proximal map.

use std::cell::RefCell;

use crate::error::{finite, invalid};
use crate::{FitError, Penalty, ProximalPenalty};

/// Complete penalty `l1 * sum_j |theta[j]| + fusion * sum_{i<j} |theta[i]-theta[j]|`.
///
/// Both strengths are finite and nonnegative, checked on use. Each unordered
/// pair contributes once. Fusion concerns signed optimization-scale values:
/// opposite signs are distinct even when their magnitudes agree. Intercepts
/// are unpenalized. Empty coefficient vectors are valid.
///
/// The exact prox sorts signed values, pools adjacent violations of their
/// rank-shifted values, and soft-thresholds the pooled levels. Value and prox
/// take `O(p log p)` time and reuse `O(p)` CPU scratch storage. There is no
/// explicit pairwise graph. [`crate::ProximalGradient`] fits this penalty
/// through typed and runtime composition; [`crate::Hybrid`] additionally
/// performs exact Gaussian cluster-coordinate sweeps.
///
/// ```
/// use shrinkage::{ClusteredLasso, Penalty, ProximalPenalty};
/// let penalty = ClusteredLasso::new(0.1, 0.2);
/// assert!((penalty.value(&[-2.0, 2.0])? - 1.2).abs() < 1e-14);
/// let mut output = [0.0; 2];
/// penalty.prox(&[-2.0, 2.0], 1.0, &mut output)?;
/// assert_eq!(output, [-1.7, 1.7]);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
#[derive(Clone, Debug)]
pub struct ClusteredLasso {
    l1: f64,
    fusion: f64,
    scratch: RefCell<Workspace>,
}

#[derive(Clone, Debug, Default)]
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

impl ClusteredLasso {
    /// Set separate L1 and uniform fusion strengths, checked on use.
    pub fn new(l1: f64, fusion: f64) -> Self {
        Self {
            l1,
            fusion,
            scratch: RefCell::new(Workspace::default()),
        }
    }

    /// Configured L1 strength.
    pub fn l1(&self) -> f64 {
        self.l1
    }

    /// Configured strength of each unordered pair difference.
    pub fn fusion(&self) -> f64 {
        self.fusion
    }

    pub(crate) fn sorted_bound_indices<T>(
        &self,
        length: usize,
        bound: impl Fn(usize) -> f64,
        inspect: impl FnOnce(&[usize]) -> T,
    ) -> T {
        let mut scratch = self.scratch.borrow_mut();
        scratch.order.clear();
        scratch.order.extend(0..length);
        scratch
            .order
            .sort_unstable_by(|&a, &b| bound(b).total_cmp(&bound(a)).then(a.cmp(&b)));
        inspect(&scratch.order)
    }
}

impl Workspace {
    fn sort(&mut self, input: &[f64]) {
        self.order.clear();
        self.order.extend(0..input.len());
        self.order
            .sort_unstable_by(|&a, &b| input[a].total_cmp(&input[b]).then(a.cmp(&b)));
    }
}

impl Penalty for ClusteredLasso {
    type Error = FitError;

    fn validate_input(&self, coefficients: &[f64]) -> Result<(), FitError> {
        if !self.l1.is_finite() || self.l1 < 0.0 || !self.fusion.is_finite() || self.fusion < 0.0 {
            return Err(invalid(
                "clustered-lasso strengths must be finite and nonnegative",
            ));
        }
        if coefficients.iter().any(|x| !x.is_finite()) {
            return Err(invalid("clustered-lasso coefficients must be finite"));
        }
        Ok(())
    }

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.validate_input(coefficients)?;
        let mut value = 0.0;
        if self.l1 != 0.0 {
            value = coefficients.iter().map(|x| self.l1 * x.abs()).sum();
        }
        if self.fusion != 0.0 && coefficients.len() > 1 {
            let mut scratch = self.scratch.borrow_mut();
            scratch.sort(coefficients);
            // Adjacent gaps avoid cancellation between signed rank sums.
            for (index, pair) in scratch.order.windows(2).enumerate() {
                let rank = (index + 1) as f64;
                let gap = coefficients[pair[1]] - coefficients[pair[0]];
                if gap != 0.0 {
                    value += (self.fusion * gap) * rank * (coefficients.len() as f64 - rank);
                }
            }
        }
        Ok(finite(value, "computing the clustered-lasso penalty", 0)?)
    }
}

impl ProximalPenalty for ClusteredLasso {
    fn gap_safe(&self) -> Option<&dyn crate::screening::GapSafePenalty> {
        Some(self)
    }

    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.prox_impl(input, None, step, output)
    }
}

impl ClusteredLasso {
    pub(crate) fn prox_screened(
        &self,
        input: &[f64],
        zeroed: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        if input.len() != zeroed.len() {
            return Err(invalid("zero mask length must match the coefficients"));
        }
        self.prox_impl(input, Some(zeroed), step, output)
    }

    fn prox_impl(
        &self,
        input: &[f64],
        zeroed: Option<&[bool]>,
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
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
        let fixed = zeroed.map_or(0, |mask| mask.iter().filter(|&&value| value).count());
        output.fill(0.0);
        if fixed == input.len() {
            return Ok(());
        }
        // Each pair between a retained coefficient and a fixed zero becomes
        // an additional absolute-value term on the retained coefficient.
        let l1_strength = finite(
            self.l1 + self.fusion * fixed as f64,
            "restricting the clustered-lasso L1 strength",
            0,
        )?;
        let l1 = finite(
            step * l1_strength,
            "scaling the clustered-lasso L1 strength",
            0,
        )?;
        let fusion = finite(
            step * self.fusion,
            "scaling the clustered-lasso fusion strength",
            0,
        )?;
        if (l1_strength > 0.0 && l1 == 0.0) || (self.fusion > 0.0 && fusion == 0.0) {
            return Err(FitError::NumericalFailure {
                operation: "scaling clustered-lasso strengths without underflow",
                iteration: 0,
            });
        }
        let mut scratch = self.scratch.borrow_mut();
        scratch.sort(input);
        if let Some(mask) = zeroed {
            scratch.order.retain(|&index| !mask[index]);
        }
        let Workspace { order, blocks } = &mut *scratch;
        blocks.clear();
        for (rank, &index) in order.iter().enumerate() {
            let shift = finite(
                fusion * (order.len() as f64 - 1.0 - 2.0 * rank as f64),
                "computing a clustered-lasso rank shift",
                0,
            )?;
            blocks.push(Block {
                start: rank,
                end: rank + 1,
                mean: finite(
                    input[index] + shift,
                    "shifting clustered-lasso proximal input",
                    0,
                )?,
            });
            while blocks.len() > 1 {
                let last = blocks.len() - 1;
                let (left, right) = (blocks[last - 1], blocks[last]);
                if left.mean <= right.mean {
                    break;
                }
                let fraction = (right.end - right.start) as f64 / (right.end - left.start) as f64;
                blocks[last - 1] = Block {
                    start: left.start,
                    end: right.end,
                    mean: finite(
                        left.mean * (1.0 - fraction) + right.mean * fraction,
                        "pooling clustered-lasso proximal blocks",
                        0,
                    )?,
                };
                blocks.pop();
            }
        }
        for block in blocks.iter() {
            let level = (block.mean.abs() - l1).max(0.0).copysign(block.mean);
            for &index in &order[block.start..block.end] {
                output[index] = if level == 0.0 { 0.0 } else { level };
            }
        }
        Ok(())
    }
}
