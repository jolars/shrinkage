//! Exact signed blocks and scalar restrictions of the averaged objective.

use lazymatrix::{LazyMatrix, RawColumns};

use crate::error::finite;
use crate::{ClusteredLasso, FitError};

#[derive(Default)]
pub(super) struct Block {
    id: usize,
    pub(super) members: Vec<usize>,
    pub(super) value: f64,
    pub(super) column: Vec<f64>,
    pub(super) curvature: f64,
}

#[derive(Default)]
struct Sum {
    sum: f64,
    correction: f64,
}

impl Sum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn total(&self) -> f64 {
        self.sum + self.correction
    }
}

fn numerical<E>(operation: &'static str, iteration: usize) -> FitError<E> {
    FitError::NumericalFailure {
        operation,
        iteration,
    }
}

pub(super) fn curvature<E>(column: &[f64], iteration: usize) -> Result<f64, FitError<E>> {
    for &value in column {
        finite(value, "forming a cluster aggregate", iteration)?;
    }
    let q = finite(
        column.iter().map(|x| x * x / column.len() as f64).sum(),
        "computing averaged cluster curvature",
        iteration,
    )?;
    if q == 0.0 && column.iter().any(|&x| x != 0.0) {
        return Err(numerical(
            "computing cluster curvature without underflow",
            iteration,
        ));
    }
    Ok(q)
}

impl super::Partition<ClusteredLasso> for State {
    fn new(n: usize, p: usize) -> Self {
        Self::new(n, p)
    }

    fn rebuild<M: RawColumns<f64> + lazymatrix::MatrixErrorType>(
        &mut self,
        matrix: &LazyMatrix<M>,
        coefficients: &[f64],
        residual: &[f64],
        _: &ClusteredLasso,
        iteration: usize,
    ) -> Result<(), FitError<M::Error>> {
        self.rebuild(matrix, coefficients, residual, iteration)
    }

    fn sweep<E>(
        &mut self,
        coefficients: &mut [f64],
        intercept: &mut f64,
        fit_intercept: bool,
        penalty: &ClusteredLasso,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        self.sweep(coefficients, intercept, fit_intercept, penalty, iteration)
    }

    fn labels(&self) -> &[usize] {
        &self.labels
    }

    fn values(&self) -> Vec<f64> {
        self.blocks.iter().map(|block| block.value).collect()
    }
}

/// Minimize the original scalar restriction, retaining median intervals.
pub(super) fn scalar_minimum(
    q: f64,
    b: f64,
    current: f64,
    knots: &mut Vec<(f64, f64)>,
    iteration: usize,
) -> Result<f64, FitError> {
    if !q.is_finite() || q < 0.0 || !b.is_finite() || (q == 0.0 && b != 0.0) {
        return Err(numerical("forming a cluster scalar subproblem", iteration));
    }
    for &(knot, weight) in knots.iter() {
        finite(knot, "forming cluster knots", iteration)?;
        finite(weight, "forming cluster knot weights", iteration)?;
    }
    knots.retain(|&(_, w)| w > 0.0);
    let mut count = 0;
    let mut weight = Sum::default();
    for index in 0..knots.len() {
        let (knot, w) = knots[index];
        if count > 0 && knots[count - 1].0 == knot {
            weight.add(w);
            knots[count - 1].1 = weight.total();
        } else {
            weight = Sum::default();
            weight.add(w);
            knots[count] = (knot, w);
            count += 1;
        }
    }
    knots.truncate(count);
    let mut total = Sum::default();
    for &(_, weight) in knots.iter() {
        total.add(weight);
    }
    let total = finite(total.total(), "summing cluster knot weights", iteration)?;
    let mut prefix = Sum::default();
    if q == 0.0 {
        if total == 0.0 {
            return Ok(current);
        }
        for (index, &(knot, weight)) in knots.iter().enumerate() {
            prefix.add(weight);
            if prefix.total() > total / 2.0 {
                return Ok(knot);
            }
            if prefix.total() == total / 2.0 {
                return Ok(current.clamp(knot, knots[index + 1].0));
            }
        }
        return Err(numerical("locating a cluster weighted median", iteration));
    }
    for &(knot, weight) in knots.iter() {
        let root = finite(
            (b + total - 2.0 * prefix.total()) / q,
            "minimizing a cluster coordinate",
            iteration,
        )?;
        if root <= knot {
            return Ok(root);
        }
        prefix.add(weight);
        let right = finite(
            q * knot - b + 2.0 * prefix.total() - total,
            "checking a cluster knot derivative",
            iteration,
        )?;
        if right >= 0.0 {
            return Ok(knot);
        }
    }
    Ok(finite(
        (b - total) / q,
        "minimizing a cluster coordinate",
        iteration,
    )?)
}

pub(super) struct State {
    pub(super) blocks: Vec<Block>,
    pub(super) labels: Vec<usize>,
    pub(super) residual: Vec<f64>,
    positions: Vec<usize>,
    order: Vec<usize>,
    sweep_order: Vec<usize>,
    knots: Vec<(f64, f64)>,
    spare: Vec<Block>,
}

impl State {
    pub(super) fn new(n: usize, p: usize) -> Self {
        Self {
            blocks: Vec::new(),
            labels: vec![usize::MAX; p],
            residual: vec![0.0; n],
            positions: vec![usize::MAX; p],
            order: Vec::with_capacity(p),
            sweep_order: Vec::with_capacity(p),
            knots: Vec::with_capacity(p + 1),
            spare: Vec::new(),
        }
    }

    pub(super) fn rebuild<M: RawColumns<f64> + lazymatrix::MatrixErrorType>(
        &mut self,
        matrix: &LazyMatrix<M>,
        coefficients: &[f64],
        residual: &[f64],
        iteration: usize,
    ) -> Result<(), FitError<M::Error>> {
        self.spare.append(&mut self.blocks);
        self.positions.fill(usize::MAX);
        self.order.clear();
        self.order.extend(0..coefficients.len());
        self.order.sort_unstable_by(|&a, &b| {
            let left = if coefficients[a] == 0.0 {
                0.0
            } else {
                coefficients[a]
            };
            let right = if coefficients[b] == 0.0 {
                0.0
            } else {
                coefficients[b]
            };
            left.total_cmp(&right).then(a.cmp(&b))
        });
        for &member in &self.order {
            let value = if coefficients[member] == 0.0 {
                0.0
            } else {
                coefficients[member]
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
            self.blocks[label].members.push(member);
            self.labels[member] = label;
        }
        // Canonical feature order makes rebuild arithmetic independent of the
        // previous sequence of crossings and aggregate additions.
        for j in 0..coefficients.len() {
            matrix
                .column(j)
                .scaled_add_to(1.0, &mut self.blocks[self.labels[j]].column);
        }
        for block in &mut self.blocks {
            block.curvature = curvature(&block.column, iteration)?;
        }
        for block in &mut self.spare {
            block.members.clear();
            // Retired large memberships must not accumulate across changing
            // partitions; small buffers remain useful for future singletons.
            block.members.shrink_to(8);
        }
        self.residual.copy_from_slice(residual);
        Ok(())
    }

    pub(super) fn sweep<E>(
        &mut self,
        coefficients: &mut [f64],
        intercept: &mut f64,
        fit_intercept: bool,
        penalty: &ClusteredLasso,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        self.sweep_order.clear();
        self.sweep_order
            .extend(self.blocks.iter().map(|block| block.id));
        for visit in 0..self.sweep_order.len() {
            let index = self.positions[self.sweep_order[visit]];
            if index == usize::MAX {
                continue;
            }
            let block = &self.blocks[index];
            let n = self.residual.len() as f64;
            let b = finite(
                block
                    .column
                    .iter()
                    .zip(&self.residual)
                    .map(|(&a, &r)| a * (r + a * block.value) / n)
                    .sum(),
                "forming a cluster linear term",
                iteration,
            )?;
            self.knots.clear();
            let mass = block.members.len() as f64;
            let mut inserted_zero = false;
            for (other_index, other) in self.blocks.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                if !inserted_zero && other.value >= 0.0 {
                    self.knots.push((0.0, penalty.l1() * mass));
                    inserted_zero = true;
                }
                self.knots.push((
                    other.value,
                    penalty.fusion() * mass * other.members.len() as f64,
                ));
            }
            if !inserted_zero {
                self.knots.push((0.0, penalty.l1() * mass));
            }
            let value = scalar_minimum(block.curvature, b, block.value, &mut self.knots, iteration)
                .map_err(|error| FitError::from_component(error, iteration))?;
            let value = if value == 0.0 { 0.0 } else { value };
            let delta = finite(
                value - block.value,
                "moving a cluster coordinate",
                iteration,
            )?;
            if delta == 0.0 {
                continue;
            }
            for (&a, r) in block.column.iter().zip(&mut self.residual) {
                *r = finite(*r - a * delta, "updating cluster residuals", iteration)?;
            }
            for &j in &block.members {
                coefficients[j] = value;
            }
            self.move_block(index, value, iteration)?;
        }
        if fit_intercept {
            let delta = finite(
                self.residual
                    .iter()
                    .map(|r| r / self.residual.len() as f64)
                    .sum(),
                "minimizing the hybrid intercept",
                iteration,
            )?;
            *intercept = finite(
                *intercept + delta,
                "updating the hybrid intercept",
                iteration,
            )?;
            for r in &mut self.residual {
                *r = finite(*r - delta, "updating intercept residuals", iteration)?;
            }
        }
        Ok(())
    }

    fn move_block<E>(
        &mut self,
        index: usize,
        value: f64,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        let mut moved = self.blocks.remove(index);
        moved.value = value;
        let position = self.blocks.partition_point(|block| block.value < value);
        if self
            .blocks
            .get(position)
            .is_some_and(|block| block.value == value)
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
            moved.curvature = curvature(&moved.column, iteration)?;
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
