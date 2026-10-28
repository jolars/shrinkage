//! Signed magnitude blocks and exact rank-dependent scalar restrictions.

#[cfg(test)]
mod tests;

use lazymatrix::{LazyMatrix, RawColumns};

use super::Partition;
use super::cluster::Sum;
use super::quadratic::Weights;
use crate::error::finite;
use crate::solver::observer::notify_inner;
use crate::{FitError, Slope};
use crate::{SolverObserver, StepKind};

#[derive(Default)]
struct Block {
    id: usize,
    members: Vec<usize>,
    value: f64,
    column: Vec<f64>,
    curvature: f64,
}

pub(super) struct State {
    blocks: Vec<Block>,
    labels: Vec<usize>,
    signs: Vec<f64>,
    residual: Vec<f64>,
    weights: Weights,
    positions: Vec<usize>,
    order: Vec<usize>,
    sweep_order: Vec<usize>,
    others: Vec<(f64, usize)>,
    spare: Vec<Block>,
}

/// Minimize the complete scalar restriction, including rank changes.
/// Other positive magnitudes are strictly decreasing and carry feature counts.
fn scalar_minimum(
    q: f64,
    b: f64,
    current: f64,
    mass: usize,
    others: &[(f64, usize)],
    weights: &[f64],
    iteration: usize,
) -> Result<f64, FitError> {
    if !q.is_finite() || q < 0.0 || !b.is_finite() || (q == 0.0 && b != 0.0) {
        return Err(FitError::NumericalFailure {
            operation: "forming a SLOPE scalar subproblem",
            iteration,
        });
    }
    let mut rank: usize = others.iter().map(|&(_, count)| count).sum();
    let mut sum = Sum::default();
    for &weight in &weights[rank..rank + mass] {
        sum.add(weight);
    }
    let mut weight = finite(sum.total(), "summing SLOPE cluster weights", iteration)?;
    let target = b.abs();
    if q > 0.0 && target <= weight || q == 0.0 && weight > 0.0 {
        return Ok(0.0);
    }
    let mut lower = 0.0_f64;
    for &(knot, count) in others.iter().rev() {
        if q > 0.0 {
            let root = (target - weight) / q;
            if root <= knot {
                return Ok(root.max(lower).copysign(b));
            }
        }
        // Nonnegative window increments avoid subtracting nearly equal prefix
        // sums, which could erase small trailing weights after a large weight.
        for j in rank - count..rank {
            sum.add(weights[j] - weights[j + mass]);
        }
        rank -= count;
        weight = finite(sum.total(), "updating SLOPE rank weights", iteration)?;
        if q == 0.0 {
            if weight > 0.0 {
                return Ok(current.min(knot));
            }
        } else if (target - weight) / q <= knot {
            return Ok(knot.copysign(b));
        }
        lower = knot;
    }
    if q == 0.0 {
        return Ok(current);
    }
    Ok(finite(
        ((target - weight) / q).max(lower).copysign(b),
        "minimizing a SLOPE cluster coordinate",
        iteration,
    )?)
}

impl Partition<Slope> for State {
    fn new(n: usize, p: usize) -> Self {
        Self {
            blocks: Vec::new(),
            labels: vec![usize::MAX; p],
            signs: vec![0.0; p],
            residual: vec![0.0; n],
            weights: Weights::default(),
            positions: vec![usize::MAX; p],
            order: Vec::with_capacity(p),
            sweep_order: Vec::with_capacity(p),
            others: Vec::with_capacity(p),
            spare: Vec::new(),
        }
    }

    fn set_weights(&mut self, weights: &[f64]) {
        self.weights.set(weights);
    }

    fn rebuild<M: RawColumns<f64> + lazymatrix::MatrixErrorType>(
        &mut self,
        matrix: &LazyMatrix<M>,
        coefficients: &[f64],
        residual: &[f64],
        _: &Slope,
        iteration: usize,
    ) -> Result<(), FitError<M::Error>> {
        self.spare.append(&mut self.blocks);
        self.positions.fill(usize::MAX);
        self.order.clear();
        self.order.extend(0..coefficients.len());
        self.order.sort_unstable_by(|&a, &b| {
            coefficients[b]
                .abs()
                .total_cmp(&coefficients[a].abs())
                .then(a.cmp(&b))
        });
        for &j in &self.order {
            let value = coefficients[j].abs();
            self.signs[j] = if value == 0.0 {
                0.0
            } else {
                coefficients[j].signum()
            };
            if self.blocks.last().is_none_or(|block| block.value != value) {
                let mut block = self.spare.pop().unwrap_or_default();
                block.id = self.blocks.len();
                block.value = value;
                block.members.clear();
                block.column.resize(residual.len(), 0.0);
                block.column.fill(0.0);
                self.positions[block.id] = self.blocks.len();
                self.blocks.push(block);
            }
            let label = self.blocks.len() - 1;
            self.blocks[label].members.push(j);
            self.labels[j] = label;
        }
        for (j, &sign) in self.signs.iter().enumerate() {
            if sign != 0.0 {
                matrix
                    .column(j)
                    .scaled_add_to(sign, &mut self.blocks[self.labels[j]].column);
            }
        }
        for block in &mut self.blocks {
            block.curvature = self.weights.curvature(&block.column, iteration)?;
        }
        for block in &mut self.spare {
            block.members.clear();
            block.members.shrink_to(8);
        }
        self.residual.copy_from_slice(residual);
        Ok(())
    }

    fn sweep_observed<E, T: SolverObserver + ?Sized>(
        &mut self,
        coefficients: &mut [f64],
        intercept: &mut f64,
        fit_intercept: bool,
        penalty: &Slope,
        iteration: usize,
        observer: &mut T,
    ) -> Result<(), FitError<E>> {
        self.sweep_order.clear();
        self.sweep_order.extend(
            self.blocks
                .iter()
                .filter(|block| block.value > 0.0)
                .map(|block| block.id),
        );
        for visit in 0..self.sweep_order.len() {
            let index = self.positions[self.sweep_order[visit]];
            if index == usize::MAX || self.blocks[index].value == 0.0 {
                continue;
            }
            let block = &self.blocks[index];
            let linear =
                self.weights
                    .linear(&block.column, &self.residual, block.value, iteration)?;
            self.others.clear();
            self.others.extend(
                self.blocks
                    .iter()
                    .enumerate()
                    .filter(|(other_index, other)| *other_index != index && other.value > 0.0)
                    .map(|(_, other)| (other.value, other.members.len())),
            );
            let value = scalar_minimum(
                block.curvature,
                linear,
                block.value,
                block.members.len(),
                &self.others,
                penalty.weights(),
                iteration,
            )
            .map_err(|error| FitError::from_component(error, iteration))?;
            let delta = finite(value - block.value, "moving a SLOPE cluster", iteration)?;
            if delta == 0.0 {
                continue;
            }
            self.weights
                .subtract(&block.column, &mut self.residual, delta, iteration)?;
            for &j in &block.members {
                coefficients[j] = if value == 0.0 {
                    0.0
                } else {
                    self.signs[j] * value
                };
            }
            self.move_block(index, value, iteration)?;
            notify_inner(
                observer,
                coefficients,
                *intercept,
                iteration + 1,
                visit + 1,
                StepKind::Block { index: visit },
            );
        }
        if fit_intercept {
            let previous = *intercept;
            self.weights
                .update_intercept(&mut self.residual, intercept, iteration)?;
            if *intercept != previous {
                notify_inner(
                    observer,
                    coefficients,
                    *intercept,
                    iteration + 1,
                    self.sweep_order.len() + 1,
                    StepKind::Intercept,
                );
            }
        }
        Ok(())
    }

    fn labels(&self) -> &[usize] {
        &self.labels
    }

    fn values(&self) -> Vec<f64> {
        self.blocks.iter().map(|block| block.value).collect()
    }
}

impl State {
    fn move_block<E>(
        &mut self,
        index: usize,
        value: f64,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        let mut moved = self.blocks.remove(index);
        moved.value = value.abs();
        if value == 0.0 {
            for &j in &moved.members {
                self.signs[j] = 0.0;
            }
            moved.column.fill(0.0);
            moved.curvature = 0.0;
        } else if value < 0.0 {
            for &j in &moved.members {
                self.signs[j] = -self.signs[j];
            }
            for a in &mut moved.column {
                *a = -*a;
            }
        }
        let position = self
            .blocks
            .partition_point(|block| block.value > moved.value);
        if self
            .blocks
            .get(position)
            .is_some_and(|block| block.value == moved.value)
        {
            let mut other = self.blocks.remove(position);
            self.positions[other.id] = usize::MAX;
            if other.members.capacity() > moved.members.capacity() {
                std::mem::swap(&mut moved.members, &mut other.members);
            }
            moved.members.extend_from_slice(&other.members);
            moved.members.sort_unstable();
            for (a, b) in moved.column.iter_mut().zip(&other.column) {
                *a += b;
            }
            moved.curvature = self.weights.curvature(&moved.column, iteration)?;
            other.members.clear();
            other.members.shrink_to(8);
            self.spare.push(other);
        }
        self.blocks.insert(position, moved);
        for (label, block) in self.blocks.iter().enumerate().skip(index.min(position)) {
            self.positions[block.id] = label;
            for &j in &block.members {
                self.labels[j] = label;
            }
        }
        Ok(())
    }
}
