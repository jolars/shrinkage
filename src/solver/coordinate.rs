use super::observer::{NoObserver, notify};
use super::working_set::{WorkingSetPolicy, Workspace as WorkingSet};
use super::{SolverObserver, StepKind};
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
    pub working_set: crate::WorkingSetDiagnostics,
}

struct Diagnostics {
    objective: f64,
    kkt_violation: f64,
    dual_objective: f64,
    duality_gap: f64,
}

fn diagnostics<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    y: &[f64],
    parameters: (&[f64], f64, &mut [f64], &mut [f64], &mut [f64]),
    residual: &Residual,
    summaries: &[ColumnSummary],
    components: (Gaussian<'_>, L1),
    iteration: usize,
) -> Result<Diagnostics, NumericalFailure> {
    let (datafit, penalty) = components;
    let (coefficients, intercept, dual_certificate, correlations, primal_correlations) = parameters;
    let kkt_violation = kkt_violation(
        matrix,
        coefficients,
        residual,
        summaries,
        (datafit, penalty),
        iteration,
        primal_correlations,
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
    for (&base, dual) in residual.base.iter().zip(dual_certificate.iter_mut()) {
        *dual = finite(
            (base + residual.offset - mean) / n,
            "constructing a dual certificate",
            iteration,
        )?;
    }
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
    for (j, entry) in correlations.iter_mut().enumerate() {
        let correlation = finite(
            matrix.column(j).dot_with_sum(dual_certificate, dual_sum),
            "checking dual feasibility",
            iteration,
        )?;
        max_correlation = max_correlation.max(correlation.abs());
        *entry = correlation;
    }
    let factor = if max_correlation > 0.0 {
        (penalty.lambda / max_correlation).min(1.0) * (1.0 - 16.0 * f64::EPSILON)
    } else {
        1.0
    };
    for value in dual_certificate.iter_mut() {
        *value = finite(*value * factor, "scaling a dual certificate", iteration)?;
    }
    let inner_product = finite(
        y.iter()
            .zip(dual_certificate.iter())
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
    for (&base, &dual) in residual.base.iter().zip(dual_certificate.iter()) {
        let difference = finite(
            base + residual.offset - n * dual,
            "computing the duality gap",
            iteration,
        )?;
        duality_gap += 0.5 * (difference / n.sqrt()).powi(2);
    }
    for (&coefficient, &correlation) in coefficients.iter().zip(correlations.iter()) {
        let slack = penalty.lambda * coefficient.abs() - coefficient * factor * correlation;
        duality_gap += slack.max(0.0);
    }
    if datafit.fit_intercept {
        let scaled_sum: f64 = dual_certificate.iter().sum();
        duality_gap -= intercept * scaled_sum;
    }
    duality_gap = finite(duality_gap.max(0.0), "computing the duality gap", iteration)?;
    for correlation in correlations.iter_mut() {
        *correlation *= factor;
    }
    Ok(Diagnostics {
        objective,
        kkt_violation,
        dual_objective,
        duality_gap,
    })
}

fn stopping_value(diagnostics: &Diagnostics, criterion: StoppingCriterion) -> f64 {
    match criterion {
        StoppingCriterion::DualityGap { .. } => diagnostics.duality_gap,
        StoppingCriterion::KktViolation { .. } => diagnostics.kkt_violation,
        StoppingCriterion::ProximalGradientMapping { .. } => {
            unreachable!("criterion was validated before iteration")
        }
    }
}

fn kkt_violation<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    coefficients: &[f64],
    residual: &Residual,
    summaries: &[ColumnSummary],
    components: (Gaussian<'_>, L1),
    iteration: usize,
    correlations: &mut [f64],
) -> Result<f64, NumericalFailure> {
    let (datafit, penalty) = components;
    let mut violation = if datafit.fit_intercept {
        residual.mean(iteration)?.abs()
    } else {
        0.0
    };
    let n = matrix.nrows() as f64;
    for (j, &coefficient) in coefficients.iter().enumerate() {
        let correlation = residual.dot(&matrix.column(j), &summaries[j], iteration)? / n;
        correlations[j] = correlation;
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

fn coordinate_violation(coefficient: f64, correlation: f64, lambda: f64) -> f64 {
    if coefficient == 0.0 {
        (correlation.abs() - lambda).max(0.0)
    } else {
        (correlation - lambda * coefficient.signum()).abs()
    }
}

pub(crate) struct Workspace {
    summaries: Vec<ColumnSummary>,
    coefficients: Vec<f64>,
    intercept: f64,
    residual: Residual,
    reference_loss: f64,
    dual_certificate: Vec<f64>,
    dual_correlations: Vec<f64>,
    correlations: Vec<f64>,
    working: WorkingSet,
    path: bool,
    screened: Vec<bool>,
}

impl Workspace {
    pub(crate) fn deviance(&self) -> Result<f64, NumericalFailure> {
        finite(
            2.0 * self.residual.loss(0)?,
            "computing Gaussian path deviance",
            0,
        )
    }

    pub(crate) fn new<M: RawColumns<f64>>(
        matrix: &LazyMatrix<M>,
        datafit: Gaussian<'_>,
    ) -> Result<Self, NumericalFailure> {
        let n = matrix.nrows() as f64;
        let intercept = if datafit.fit_intercept {
            finite(
                datafit.response.iter().map(|value| value / n).sum(),
                "computing the response mean",
                0,
            )?
        } else {
            0.0
        };
        let residual = Residual::new(datafit.response, intercept)?;
        let reference_loss = residual.loss(0)?;
        Ok(Self {
            summaries: (0..matrix.ncols())
                .map(|j| ColumnSummary::new(&matrix.column(j)))
                .collect::<Result<_, _>>()?,
            coefficients: vec![0.0; matrix.ncols()],
            intercept,
            residual,
            reference_loss,
            dual_certificate: vec![0.0; matrix.nrows()],
            dual_correlations: vec![0.0; matrix.ncols()],
            correlations: vec![0.0; matrix.ncols()],
            working: WorkingSet::new(matrix.ncols()),
            path: false,
            screened: vec![false; matrix.ncols()],
        })
    }

    fn screen(&mut self, penalty: L1, options: &CoordinateDescent, check: &Diagnostics) {
        // The dual objective is n-strongly concave in the unscaled certificate.
        // Inflate the gap and the strict screening bound to retain borderline
        // columns under floating-point rounding.
        let guard = 64.0
            * f64::EPSILON
            * (check.objective.abs() + check.dual_objective.abs() + self.reference_loss);
        let radius = (2.0 * ((check.duality_gap + guard) / self.residual.base.len() as f64)).sqrt();
        for j in 0..self.coefficients.len() {
            let nonzero = self.coefficients[j] != 0.0;
            let bound =
                self.dual_correlations[j].abs() + self.summaries[j].norm_squared.sqrt() * radius;
            let slack = 64.0 * f64::EPSILON * (bound.abs() + penalty.lambda);
            self.screened[j] = options.screening
                && penalty.lambda > 0.0
                && !nonzero
                && bound + slack < penalty.lambda;
            let violation = self.correlations[j].abs() > penalty.lambda;
            // Full KKT checks can restore a coordinate even when a screening
            // bound was too optimistic because of numerical rounding.
            if violation {
                self.screened[j] = false;
            }
            self.working.scores[j] =
                coordinate_violation(self.coefficients[j], self.correlations[j], penalty.lambda);
        }
    }
    fn diagnose<M: RawColumns<f64>>(
        &mut self,
        matrix: &LazyMatrix<M>,
        datafit: Gaussian<'_>,
        penalty: L1,
        iteration: usize,
    ) -> Result<Diagnostics, NumericalFailure> {
        self.residual.refresh(
            matrix,
            datafit.response,
            &self.coefficients,
            self.intercept,
            &self.summaries,
            iteration,
        )?;
        self.working.diagnostics.full_checks += 1;
        diagnostics(
            matrix,
            datafit.response,
            (
                &self.coefficients,
                self.intercept,
                &mut self.dual_certificate,
                &mut self.dual_correlations,
                &mut self.correlations,
            ),
            &self.residual,
            &self.summaries,
            (datafit, penalty),
            iteration,
        )
    }
}

pub(crate) fn solve<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    datafit: Gaussian<'_>,
    penalty: L1,
    options: &CoordinateDescent,
) -> Result<Solution, NumericalFailure> {
    solve_observed(matrix, datafit, penalty, options, None, &mut NoObserver)
}

pub(crate) fn solve_observed<M: RawColumns<f64>, T: SolverObserver + ?Sized>(
    matrix: &LazyMatrix<M>,
    datafit: Gaussian<'_>,
    penalty: L1,
    options: &CoordinateDescent,
    initial: Option<crate::InitialPoint<'_>>,
    observer: &mut T,
) -> Result<Solution, NumericalFailure> {
    let mut work = Workspace::new(matrix, datafit)?;
    if let Some(initial) = initial {
        // Public entry points validate the parameters before the numerical kernel.
        work.coefficients.copy_from_slice(initial.coefficients);
        work.intercept = initial.intercept;
        work.residual.refresh(
            matrix,
            datafit.response,
            &work.coefficients,
            work.intercept,
            &work.summaries,
            0,
        )?;
    }
    let full = CoordinateDescent {
        screening: false,
        ..*options
    };
    solve_with_workspace_observed(matrix, datafit, penalty, &full, &mut work, observer)
}

pub(crate) fn solve_with_workspace<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    datafit: Gaussian<'_>,
    penalty: L1,
    options: &CoordinateDescent,
    work: &mut Workspace,
) -> Result<Solution, NumericalFailure> {
    work.path = true;
    solve_with_workspace_observed(matrix, datafit, penalty, options, work, &mut NoObserver)
}

pub(crate) fn solve_with_workspace_observed<M: RawColumns<f64>, T: SolverObserver + ?Sized>(
    matrix: &LazyMatrix<M>,
    datafit: Gaussian<'_>,
    penalty: L1,
    options: &CoordinateDescent,
    work: &mut Workspace,
    observer: &mut T,
) -> Result<Solution, NumericalFailure> {
    let criterion = options.criterion(penalty);
    let n = matrix.nrows() as f64;
    let reference_loss = work.reference_loss;
    let threshold = match criterion {
        StoppingCriterion::DualityGap { absolute, relative } => finite(
            absolute + relative * reference_loss,
            "computing the duality-gap threshold",
            0,
        )?,
        StoppingCriterion::KktViolation { absolute } => absolute,
        StoppingCriterion::ProximalGradientMapping { .. } => unreachable!("validated criterion"),
    };
    let stationarity_tolerance = match criterion {
        StoppingCriterion::KktViolation { absolute } => absolute,
        _ => 0.0,
    };
    work.working
        .begin(options.working_set_policy, work.path, &work.coefficients);
    let mut check = work.diagnose(matrix, datafit, penalty, 0)?;
    work.screen(penalty, options, &check);
    work.working.select(
        &work.coefficients,
        &work.screened,
        false,
        false,
        stationarity_tolerance,
    );
    let mut target = stationarity_tolerance.max(0.1 * check.kkt_violation);
    let mut iterations = 0;
    let mut since_full = 0;
    let mut moved = false;
    notify(
        observer,
        &work.coefficients,
        work.intercept,
        0,
        StepKind::Initial,
    );
    while stopping_value(&check, criterion) > threshold && iterations < options.max_iterations {
        iterations += 1;
        since_full += 1;
        for &j in &work.working.active {
            work.working.diagnostics.feature_visits += 1;
            let summary = &work.summaries[j];
            if summary.norm_squared == 0.0 {
                continue;
            }
            let column = matrix.column(j);
            let curvature = summary.norm_squared / n;
            let partial = finite(
                work.residual.dot(&column, summary, iterations)? / n
                    + curvature * work.coefficients[j],
                "computing a coordinate update",
                iterations,
            )?;
            let updated = finite(
                penalty.coordinate_minimum(partial, curvature),
                "solving a coordinate subproblem",
                iterations,
            )?;
            let delta = finite(
                updated - work.coefficients[j],
                "computing a coefficient change",
                iterations,
            )?;
            if delta != 0.0 {
                work.residual
                    .subtract_column(&column, summary, delta, iterations)?;
                work.coefficients[j] = updated;
                moved = true;
                notify(
                    observer,
                    &work.coefficients,
                    work.intercept,
                    iterations,
                    StepKind::Coordinate { index: j },
                );
            }
        }
        if datafit.fit_intercept {
            let delta = work.residual.mean(iterations)?;
            work.intercept = finite(work.intercept + delta, "updating the intercept", iterations)?;
            work.residual.offset = finite(
                work.residual.offset - delta,
                "updating the intercept residual",
                iterations,
            )?;
            moved |= delta != 0.0;
            notify(
                observer,
                &work.coefficients,
                work.intercept,
                iterations,
                StepKind::Intercept,
            );
        }
        work.working.record(&work.coefficients);
        if iterations % REFRESH_INTERVAL == 0 {
            work.residual.refresh(
                matrix,
                datafit.response,
                &work.coefficients,
                work.intercept,
                &work.summaries,
                iterations,
            )?;
        }
        let disabled = work.working.diagnostics.policy == WorkingSetPolicy::Disabled;
        let local = if disabled {
            f64::INFINITY
        } else {
            let mut local = if datafit.fit_intercept {
                work.residual.mean(iterations)?.abs()
            } else {
                0.0
            };
            for &j in &work.working.active {
                let correlation =
                    work.residual
                        .dot(&matrix.column(j), &work.summaries[j], iterations)?
                        / n;
                local = local.max(coordinate_violation(
                    work.coefficients[j],
                    correlation,
                    penalty.lambda,
                ));
            }
            work.working.diagnostics.restricted_checks += 1;
            local
        };
        let discover =
            iterations == 1 && work.working.diagnostics.policy == WorkingSetPolicy::EverActive;
        if local <= target
            || !moved
            || since_full == 10
            || disabled
            || discover
            || iterations == options.max_iterations
        {
            check = work.diagnose(matrix, datafit, penalty, iterations)?;
            work.screen(penalty, options, &check);
            work.working.select(
                &work.coefficients,
                &work.screened,
                local <= target || !moved,
                !moved,
                stationarity_tolerance,
            );
            target = stationarity_tolerance.max(0.1 * check.kkt_violation);
            since_full = 0;
            moved = false;
        }
    }
    let Workspace {
        coefficients,
        intercept,
        dual_certificate,
        ..
    } = work;
    let final_value = stopping_value(&check, criterion);
    notify(
        observer,
        coefficients,
        *intercept,
        iterations,
        StepKind::Final(if final_value <= threshold {
            Termination::Converged
        } else {
            Termination::IterationLimit
        }),
    );
    Ok(Solution {
        coefficients: coefficients.clone(),
        intercept: *intercept,
        termination: if final_value <= threshold {
            Termination::Converged
        } else {
            Termination::IterationLimit
        },
        iterations,
        objective: check.objective,
        kkt_violation: check.kkt_violation,
        dual_certificate: dual_certificate.clone(),
        dual_objective: check.dual_objective,
        duality_gap: check.duality_gap,
        reference_loss,
        stopping_criterion: criterion,
        stopping_value: final_value,
        stopping_threshold: threshold,
        working_set: work.working.diagnostics.clone(),
    })
}
