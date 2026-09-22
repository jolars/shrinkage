use lazymatrix::{LazyColumn, LazyMatrix, RawColumn, RawColumns};

use super::{Lasso, Termination};

#[cfg(test)]
mod tests;

const REFRESH_INTERVAL: usize = 50;

#[derive(Debug)]
pub(super) struct NumericalFailure {
    pub operation: &'static str,
    pub iteration: usize,
}

pub(super) fn finite(
    value: f64,
    operation: &'static str,
    iteration: usize,
) -> Result<f64, NumericalFailure> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NumericalFailure {
            operation,
            iteration,
        })
    }
}

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
        let divisor = (self.base.len() as f64).sqrt();
        let mut sum = 0.0;
        for &base in &self.base {
            let value =
                finite(base + self.offset, "reconstructing residuals", iteration)? / divisor;
            sum += 0.5 * value * value;
        }
        finite(sum, "computing the loss", iteration)
    }
}

pub(super) struct Solution {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub termination: Termination,
    pub iterations: usize,
    pub objective: f64,
    pub kkt_violation: f64,
}

fn soft_threshold(value: f64, lambda: f64) -> f64 {
    if value > lambda {
        value - lambda
    } else if value < -lambda {
        value + lambda
    } else {
        0.0
    }
}

fn kkt_violation<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    coefficients: &[f64],
    residual: &Residual,
    summaries: &[ColumnSummary],
    options: &Lasso,
    iteration: usize,
) -> Result<f64, NumericalFailure> {
    let mut violation = if options.fit_intercept {
        residual.mean(iteration)?.abs()
    } else {
        0.0
    };
    let n = matrix.nrows() as f64;
    for (j, &coefficient) in coefficients.iter().enumerate() {
        let correlation = residual.dot(&matrix.column(j), &summaries[j], iteration)? / n;
        let coordinate_violation = if coefficient == 0.0 {
            (correlation.abs() - options.lambda).max(0.0)
        } else {
            finite(
                correlation - options.lambda * coefficient.signum(),
                "checking coordinate optimality",
                iteration,
            )?
            .abs()
        };
        violation = violation.max(coordinate_violation);
    }
    Ok(violation)
}

pub(super) fn solve<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    y: &[f64],
    options: &Lasso,
) -> Result<Solution, NumericalFailure> {
    let n = matrix.nrows() as f64;
    let summaries: Vec<_> = (0..matrix.ncols())
        .map(|j| ColumnSummary::new(&matrix.column(j)))
        .collect::<Result<_, _>>()?;
    let mut coefficients = vec![0.0; matrix.ncols()];
    let mut intercept = if options.fit_intercept {
        finite(
            y.iter().map(|value| value / n).sum(),
            "computing the response mean",
            0,
        )?
    } else {
        0.0
    };
    let mut residual = Residual::new(y, intercept)?;
    let mut violation = kkt_violation(matrix, &coefficients, &residual, &summaries, options, 0)?;
    let mut iterations = 0;
    while violation > options.tolerance && iterations < options.max_iterations {
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
                soft_threshold(partial, options.lambda) / curvature,
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
        if options.fit_intercept {
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
        violation = kkt_violation(
            matrix,
            &coefficients,
            &residual,
            &summaries,
            options,
            iterations,
        )?;
        if violation <= options.tolerance && !refreshed {
            residual.refresh(matrix, y, &coefficients, intercept, &summaries, iterations)?;
            violation = kkt_violation(
                matrix,
                &coefficients,
                &residual,
                &summaries,
                options,
                iterations,
            )?;
        }
    }
    let penalty = if options.lambda == 0.0 {
        0.0
    } else {
        coefficients
            .iter()
            .map(|value| options.lambda * value.abs())
            .sum()
    };
    let objective = finite(
        residual.loss(iterations)? + penalty,
        "computing the objective",
        iterations,
    )?;
    Ok(Solution {
        coefficients,
        intercept,
        termination: if violation <= options.tolerance {
            Termination::Converged
        } else {
            Termination::IterationLimit
        },
        iterations,
        objective,
        kkt_violation: violation,
    })
}
