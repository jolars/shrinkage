//! Contiguous signed segments and exact weighted-boundary restrictions.

#[cfg(test)]
mod tests;

use lazymatrix::{LazyMatrix, RawColumns};

use super::Partition;
use super::cluster::scalar_minimum;
use super::quadratic::Weights;
use crate::error::finite;
use crate::solver::observer::notify_inner;
use crate::{ChainFusedLasso, FitError};
use crate::{SolverObserver, StepKind};

#[derive(Default)]
struct Segment {
    id: usize,
    start: usize,
    end: usize,
    value: f64,
    left_weight: f64,
    right_weight: f64,
    column: Vec<f64>,
    curvature: f64,
}

pub(super) struct State {
    segments: Vec<Segment>,
    labels: Vec<usize>,
    residual: Vec<f64>,
    weights: Weights,
    positions: Vec<usize>,
    sweep_order: Vec<usize>,
    knots: Vec<(f64, f64)>,
    spare: Vec<Segment>,
    screened: Vec<bool>,
}

impl Partition<ChainFusedLasso> for State {
    fn new(n: usize, p: usize) -> Self {
        Self {
            segments: Vec::new(),
            labels: vec![usize::MAX; p],
            residual: vec![0.0; n],
            weights: Weights::default(),
            positions: vec![usize::MAX; p],
            sweep_order: Vec::with_capacity(p),
            knots: Vec::with_capacity(3),
            spare: Vec::new(),
            screened: vec![false; p],
        }
    }

    fn set_weights(&mut self, weights: &[f64]) {
        self.weights.set(weights);
    }

    fn set_screened(&mut self, zeroed: &[bool]) {
        self.screened.clear();
        self.screened.extend_from_slice(zeroed);
    }

    fn rebuild<M: RawColumns<f64> + lazymatrix::MatrixErrorType>(
        &mut self,
        matrix: &LazyMatrix<M>,
        coefficients: &[f64],
        residual: &[f64],
        penalty: &ChainFusedLasso,
        iteration: usize,
    ) -> Result<(), FitError<M::Error>> {
        self.spare.append(&mut self.segments);
        self.positions.fill(usize::MAX);
        for (j, &coefficient) in coefficients.iter().enumerate() {
            let value = if coefficient == 0.0 { 0.0 } else { coefficient };
            if self
                .segments
                .last()
                .is_none_or(|segment| segment.value != value)
                || (j > 0 && penalty.weight(j - 1) == 0.0)
            {
                let mut segment = self.spare.pop().unwrap_or_default();
                segment.id = j;
                segment.start = j;
                segment.end = j;
                segment.value = value;
                segment.left_weight = if j == 0 { 0.0 } else { penalty.weight(j - 1) };
                segment.column.resize(residual.len(), 0.0);
                segment.column.fill(0.0);
                self.positions[j] = self.segments.len();
                self.segments.push(segment);
            }
            let label = self.segments.len() - 1;
            self.labels[j] = label;
            let segment = &mut self.segments[label];
            segment.end = j + 1;
            segment.right_weight = if j + 1 == coefficients.len() {
                0.0
            } else {
                penalty.weight(j)
            };
            // Feature order fixes aggregate arithmetic independently of earlier
            // merges, including when a PG transition changes only the boundary.
            if !self.screened.get(j).copied().unwrap_or(false) {
                matrix.column(j).scaled_add_to(1.0, &mut segment.column);
            }
        }
        for segment in &mut self.segments {
            segment.curvature = self.weights.curvature(&segment.column, iteration)?;
        }
        self.residual.copy_from_slice(residual);
        Ok(())
    }

    fn sweep_observed<E, T: SolverObserver + ?Sized>(
        &mut self,
        coefficients: &mut [f64],
        intercept: &mut f64,
        fit_intercept: bool,
        penalty: &ChainFusedLasso,
        iteration: usize,
        observer: &mut T,
    ) -> Result<(), FitError<E>> {
        self.sweep_order.clear();
        self.sweep_order
            .extend(self.segments.iter().map(|segment| segment.id));
        for visit in 0..self.sweep_order.len() {
            let index = self.positions[self.sweep_order[visit]];
            if index == usize::MAX {
                continue;
            }
            let segment = &self.segments[index];
            // Ordinary zeros remain eligible for splitting at the next PG
            // transition, while this sweep retains certified zero members.
            if (segment.start..segment.end).any(|j| self.screened.get(j).copied().unwrap_or(false))
            {
                continue;
            }
            let linear =
                self.weights
                    .linear(&segment.column, &self.residual, segment.value, iteration)?;
            self.knots.clear();
            self.knots
                .push((0.0, penalty.l1() * (segment.end - segment.start) as f64));
            if index > 0 && segment.left_weight > 0.0 {
                self.knots.push((
                    self.segments[index - 1].value,
                    penalty.fusion() * segment.left_weight,
                ));
            }
            if index + 1 < self.segments.len() && segment.right_weight > 0.0 {
                self.knots.push((
                    self.segments[index + 1].value,
                    penalty.fusion() * segment.right_weight,
                ));
            }
            self.knots.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let value = scalar_minimum(
                segment.curvature,
                linear,
                segment.value,
                &mut self.knots,
                iteration,
            )
            .map_err(|error| FitError::from_component(error, iteration))?;
            let value = if value == 0.0 { 0.0 } else { value };
            let delta = finite(value - segment.value, "moving a fused segment", iteration)?;
            if delta == 0.0 {
                continue;
            }
            self.weights
                .subtract(&segment.column, &mut self.residual, delta, iteration)?;
            coefficients[segment.start..segment.end].fill(value);
            self.move_segment(index, value, iteration)?;
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
        self.segments.iter().map(|segment| segment.value).collect()
    }
}

impl State {
    fn move_segment<E>(
        &mut self,
        mut index: usize,
        value: f64,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        let left = index > 0
            && self.segments[index].left_weight > 0.0
            && self.segments[index - 1].value == value;
        let right = index + 1 < self.segments.len()
            && self.segments[index].right_weight > 0.0
            && self.segments[index + 1].value == value;
        if !left && !right {
            self.segments[index].value = value;
            return Ok(());
        }
        let mut moved = self.segments.remove(index);
        moved.value = value;
        if left {
            let other = self.segments.remove(index - 1);
            index -= 1;
            moved.start = other.start;
            moved.left_weight = other.left_weight;
            self.absorb(&mut moved, other);
        }
        if right {
            let other = self.segments.remove(index);
            moved.end = other.end;
            moved.right_weight = other.right_weight;
            self.absorb(&mut moved, other);
        }
        moved.curvature = self.weights.curvature(&moved.column, iteration)?;
        self.segments.insert(index, moved);
        for (label, segment) in self.segments.iter().enumerate().skip(index) {
            self.positions[segment.id] = label;
            self.labels[segment.start..segment.end].fill(label);
        }
        Ok(())
    }

    fn absorb(&mut self, moved: &mut Segment, other: Segment) {
        self.positions[other.id] = usize::MAX;
        for (a, &b) in moved.column.iter_mut().zip(&other.column) {
            *a += b;
        }
        self.spare.push(other);
    }
}
