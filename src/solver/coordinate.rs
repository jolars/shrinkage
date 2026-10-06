use lazymatrix::{LazyColumn, LazyMatrix, RawColumn, RawColumns};

use super::{CoordinateDescent, StoppingCriterion};
use crate::error::{NumericalFailure, finite};
use crate::{Gaussian, L1, Termination};

#[cfg(test)]
mod tests;

const REFRESH_INTERVAL: usize = 50;

struct ColumnSummary {
    raw_sum: f64,
    sum: f64,
    norm_squared: f64,
}

impl ColumnSummary {
    fn new<C: RawColumn<f64>>(column: &LazyColumn<C, f64>) -> Result<Self, NumericalFailure> {
        Ok(Self {
            raw_sum: finite(column.raw().raw_sum(), "summing a column", 0)?,
            sum: finite(column.sum(), "summing a normalized column", 0)?,
            norm_squared: finite(column.norm_squared(), "computing a column norm", 0)?,
        })
    }
}

// Centering changes every logical row. Keeping that common change separate
// makes sparse coordinate updates touch only the stored entries.
struct Residual {
    base: Vec<f64>,
    base_sum: f64,
    offset: f64,
}

impl Residual {
    fn new(y: &[f64], intercept: f64) -> Result<Self, NumericalFailure> {
        let mut residual = Self {
            base: vec![0.0; y.len()],
            base_sum: 0.0,
            offset: 0.0,
        };
        residual.reset(y, intercept, 0)?;
        Ok(residual)
    }

    fn reset(
        &mut self,
        y: &[f64],
        intercept: f64,
        iteration: usize,
    ) -> Result<(), NumericalFailure> {
        for (value, &target) in self.base.iter_mut().zip(y) {
            *value = finite(target - intercept, "initializing residuals", iteration)?;
        }
        self.offset = 0.0;
        self.base_sum = finite(self.base.iter().sum(), "summing residuals", iteration)?;
        Ok(())
    }

    fn dot<C: RawColumn<f64>>(
        &self,
        column: &LazyColumn<C, f64>,
        summary: &ColumnSummary,
        iteration: usize,
    ) -> Result<f64, NumericalFailure> {
        finite(
            column.dot_with_sum(&self.base, self.base_sum) + self.offset * summary.sum,
            "computing a residual correlation",
            iteration,
        )
    }

    fn subtract_column<C: RawColumn<f64>>(
        &mut self,
        column: &LazyColumn<C, f64>,
        summary: &ColumnSummary,
        delta: f64,
        iteration: usize,
    ) -> Result<(), NumericalFailure> {
        let multiplier = finite(
            delta / column.scale(),
            "scaling a coefficient update",
            iteration,
        )?;
        let mut valid = true;
        column.raw().for_each_stored(|i, value| {
            self.base[i] -= multiplier * value;
            valid &= self.base[i].is_finite();
        });
        if !valid {
            return Err(NumericalFailure {
                operation: "updating residuals",
                iteration,
            });
        }
        self.base_sum = finite(
            self.base_sum - multiplier * summary.raw_sum,
            "updating the residual sum",
            iteration,
        )?;
        self.offset = finite(
            self.offset + multiplier * column.center(),
            "updating the residual offset",
            iteration,
        )?;
        Ok(())
    }

    fn mean(&self, iteration: usize) -> Result<f64, NumericalFailure> {
        finite(
            self.base_sum / self.base.len() as f64 + self.offset,
            "computing the residual mean",
            iteration,
        )
    }

    fn refresh<M: RawColumns<f64>>(
        &mut self,
        matrix: &LazyMatrix<M>,
        y: &[f64],
        coefficients: &[f64],
        intercept: f64,
        summaries: &[ColumnSummary],
        iteration: usize,
    ) -> Result<(), NumericalFailure> {
        self.reset(y, intercept, iteration)?;
        for (j, &coefficient) in coefficients.iter().enumerate() {
            if coefficient != 0.0 {
                self.subtract_column(&matrix.column(j), &summaries[j], coefficient, iteration)?;
            }
        }
        // Recompute from the entries rather than carrying the update history
        // into the next convergence check.
        self.base_sum = finite(
            self.base.iter().sum(),
            "refreshing the residual sum",
            iteration,
        )?;
        Ok(())
    }

    fn loss(&self, iteration: usize) -> Result<f64, NumericalFailure> {
        Gaussian::residual_loss(
            self.base.iter().map(|base| base + self.offset),
            self.base.len(),
            iteration,
        )
    }
}

pub(crate) struct Solution {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub termination: Termination,
    pub iterations: usize,
    pub objective: f64,
    pub kkt_violation: f64,
    pub dual_certificate: Vec<f64>,
    pub dual_objective: f64,
    pub duality_gap: f64,
    pub reference_loss: f64,
    pub stopping_criterion: StoppingCriterion,
    pub stopping_value: f64,
    pub stopping_threshold: f64,
}

struct Diagnostics {
    objective: f64,
    kkt_violation: f64,
    dual_certificate: Vec<f64>,
    dual_objective: f64,
    duality_gap: f64,
}

fn diagnostics<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    y: &[f64],
    parameters: (&[f64], f64),
    residual: &Residual,
    summaries: &[ColumnSummary],
    components: (Gaussian<'_>, L1),
    iteration: usize,
) -> Result<Diagnostics, NumericalFailure> {
    let (datafit, penalty) = components;
    let (coefficients, intercept) = parameters;
    let kkt_violation = kkt_violation(
        matrix,
        coefficients,
        residual,
        summaries,
        datafit,
        penalty,
        iteration,
    )?;
    let penalty_value = finite(
        penalty.value_unchecked(coefficients),
        "computing the penalty",
        iteration,
    )?;
    let objective = finite(
        residual.loss(iteration)? + penalty_value,
        "computing the objective",
        iteration,
    )?;

    // The residual divided by n is the unconstrained dual candidate. Center
    // it when the intercept is free, then shrink it toward zero to satisfy
    // every normalized-column constraint without changing its direction.
    let n = y.len() as f64;
    let mean = if datafit.fit_intercept {
        residual.mean(iteration)?
    } else {
        0.0
    };
    let mut dual_certificate = residual
        .base
        .iter()
        .map(|&base| {
            finite(
                (base + residual.offset - mean) / n,
                "constructing a dual certificate",
                iteration,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if datafit.fit_intercept {
        let last = dual_certificate.len() - 1;
        let prefix_sum: f64 = dual_certificate[..last].iter().sum();
        dual_certificate[last] = finite(-prefix_sum, "centering a dual certificate", iteration)?;
    }
    let dual_sum = finite(
        dual_certificate.iter().sum(),
        "summing a dual certificate",
        iteration,
    )?;
    let mut max_correlation: f64 = 0.0;
    let mut correlations = Vec::with_capacity(matrix.ncols());
    for j in 0..matrix.ncols() {
        let correlation = finite(
            matrix.column(j).dot_with_sum(&dual_certificate, dual_sum),
            "checking dual feasibility",
            iteration,
        )?;
        max_correlation = max_correlation.max(correlation.abs());
        correlations.push(correlation);
    }
    let factor = if max_correlation > 0.0 {
        (penalty.lambda / max_correlation).min(1.0) * (1.0 - 16.0 * f64::EPSILON)
    } else {
        1.0
    };
    for value in &mut dual_certificate {
        *value = finite(*value * factor, "scaling a dual certificate", iteration)?;
    }
    let inner_product = finite(
        y.iter()
            .zip(&dual_certificate)
            .map(|(&target, &dual)| target * dual)
            .sum(),
        "computing the dual objective",
        iteration,
    )?;
    let squared_norm = finite(
        dual_certificate.iter().map(|value| value * value).sum(),
        "computing the dual norm",
        iteration,
    )?;
    let dual_objective = finite(
        inner_product - 0.5 * n * squared_norm,
        "computing the dual objective",
        iteration,
    )?;
    // Fenchel's identity avoids subtracting nearly equal objectives. The norm
    // and coordinatewise terms are nonnegative for an exactly feasible vector.
    let mut duality_gap = 0.0;
    for (&base, &dual) in residual.base.iter().zip(&dual_certificate) {
        let difference = finite(
            base + residual.offset - n * dual,
            "computing the duality gap",
            iteration,
        )?;
        duality_gap += 0.5 * (difference / n.sqrt()).powi(2);
    }
    for (&coefficient, &correlation) in coefficients.iter().zip(&correlations) {
        let slack = penalty.lambda * coefficient.abs() - coefficient * factor * correlation;
        duality_gap += slack.max(0.0);
    }
    if datafit.fit_intercept {
        let scaled_sum: f64 = dual_certificate.iter().sum();
        duality_gap -= intercept * scaled_sum;
    }
    duality_gap = finite(duality_gap.max(0.0), "computing the duality gap", iteration)?;
    Ok(Diagnostics {
        objective,
        kkt_violation,
        dual_certificate,
        dual_objective,
        duality_gap,
    })
}

fn stopping_value(diagnostics: &Diagnostics, criterion: StoppingCriterion) -> f64 {
    match criterion {
        StoppingCriterion::DualityGap { .. } => diagnostics.duality_gap,
        StoppingCriterion::KktViolation { .. } => diagnostics.kkt_violation,
    }
}

fn kkt_violation<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    coefficients: &[f64],
    residual: &Residual,
    summaries: &[ColumnSummary],
    datafit: Gaussian<'_>,
    penalty: L1,
    iteration: usize,
) -> Result<f64, NumericalFailure> {
    let mut violation = if datafit.fit_intercept {
        residual.mean(iteration)?.abs()
    } else {
        0.0
    };
    let n = matrix.nrows() as f64;
    for (j, &coefficient) in coefficients.iter().enumerate() {
        let correlation = residual.dot(&matrix.column(j), &summaries[j], iteration)? / n;
        let coordinate_violation = if coefficient == 0.0 {
            (correlation.abs() - penalty.lambda).max(0.0)
        } else {
            finite(
                correlation - penalty.lambda * coefficient.signum(),
                "checking coordinate optimality",
                iteration,
            )?
            .abs()
        };
        violation = violation.max(coordinate_violation);
    }
    Ok(violation)
}

pub(crate) fn solve<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    datafit: Gaussian<'_>,
    penalty: L1,
    options: &CoordinateDescent,
) -> Result<Solution, NumericalFailure> {
    let y = datafit.response;
    let criterion = options.criterion(penalty);
    let n = matrix.nrows() as f64;
    let summaries: Vec<_> = (0..matrix.ncols())
        .map(|j| ColumnSummary::new(&matrix.column(j)))
        .collect::<Result<_, _>>()?;
    let mut coefficients = vec![0.0; matrix.ncols()];
    let mut intercept = if datafit.fit_intercept {
        finite(
            y.iter().map(|value| value / n).sum(),
            "computing the response mean",
            0,
        )?
    } else {
        0.0
    };
    let mut residual = Residual::new(y, intercept)?;
    let reference_loss = residual.loss(0)?;
    let threshold = match criterion {
        StoppingCriterion::DualityGap { absolute, relative } => finite(
            absolute + relative * reference_loss,
            "computing the duality-gap threshold",
            0,
        )?,
        StoppingCriterion::KktViolation { absolute } => absolute,
    };
    let mut check = diagnostics(
        matrix,
        y,
        (&coefficients, intercept),
        &residual,
        &summaries,
        (datafit, penalty),
        0,
    )?;
    let mut iterations = 0;
    while stopping_value(&check, criterion) > threshold && iterations < options.max_iterations {
        iterations += 1;
        for (j, coefficient) in coefficients.iter_mut().enumerate() {
            let summary = &summaries[j];
            if summary.norm_squared == 0.0 {
                continue;
            }
            let column = matrix.column(j);
            let curvature = summary.norm_squared / n;
            let partial = finite(
                residual.dot(&column, summary, iterations)? / n + curvature * *coefficient,
                "computing a coordinate update",
                iterations,
            )?;
            let updated = finite(
                penalty.coordinate_minimum(partial, curvature),
                "solving a coordinate subproblem",
                iterations,
            )?;
            let delta = finite(
                updated - *coefficient,
                "computing a coefficient change",
                iterations,
            )?;
            if delta != 0.0 {
                residual.subtract_column(&column, summary, delta, iterations)?;
                *coefficient = updated;
            }
        }
        if datafit.fit_intercept {
            let delta = residual.mean(iterations)?;
            intercept = finite(intercept + delta, "updating the intercept", iterations)?;
            residual.offset = finite(
                residual.offset - delta,
                "updating the intercept residual",
                iterations,
            )?;
        }
        let refreshed = iterations % REFRESH_INTERVAL == 0 || iterations == options.max_iterations;
        if refreshed {
            residual.refresh(matrix, y, &coefficients, intercept, &summaries, iterations)?;
        }
        check = diagnostics(
            matrix,
            y,
            (&coefficients, intercept),
            &residual,
            &summaries,
            (datafit, penalty),
            iterations,
        )?;
        if stopping_value(&check, criterion) <= threshold && !refreshed {
            residual.refresh(matrix, y, &coefficients, intercept, &summaries, iterations)?;
            check = diagnostics(
                matrix,
                y,
                (&coefficients, intercept),
                &residual,
                &summaries,
                (datafit, penalty),
                iterations,
            )?;
        }
    }
    let final_value = stopping_value(&check, criterion);
    Ok(Solution {
        coefficients,
        intercept,
        termination: if final_value <= threshold {
            Termination::Converged
        } else {
            Termination::IterationLimit
        },
        iterations,
        objective: check.objective,
        kkt_violation: check.kkt_violation,
        dual_certificate: check.dual_certificate,
        dual_objective: check.dual_objective,
        duality_gap: check.duality_gap,
        reference_loss,
        stopping_criterion: criterion,
        stopping_value: final_value,
        stopping_threshold: threshold,
    })
}
