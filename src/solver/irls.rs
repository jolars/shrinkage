//! Quadratic loss models solved by cyclic coordinate descent.

use super::observer::{NoObserver, notify, notify_inner};
use super::working_set::{WorkingSetPolicy, Workspace as WorkingSet};
use super::{ObservedSolver, SolverObserver, StepKind};

use lazymatrix::{ColumnStats, LazyMatrix, MatrixErrorType, RawColumn, RawColumns};

use crate::error::{finite, invalid, unsupported};
use crate::{
    CoordinatePenalty, DiagonalHessianDatafit, FitError, IrlsFit, Problem, Solver,
    StoppingCriterion, Termination,
};

/// Iteratively reweighted least squares with coordinate-descent inner solves.
///
/// Fits convex losses with diagonal predictor Hessians and separable convex
/// penalties. Backtracking evaluates the original penalized objective. The
/// quadratic model floors predictor curvature at `1e-12 / nobs`, leaving the
/// original loss, gradient, and convergence check unchanged. Working sets use
/// restricted checks between complete KKT checks. No safe screening or dual
/// certificate is provided.
///
/// Solver workspace is `O(nobs + nfeatures)`. Weighted column norms use
/// LazyMatrix's `O(nobs + nnz)` operation once per column per outer iteration.
/// Noncanonical sparse columns can require temporary `O(nobs)` scratch there.
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::Mat;
/// use shrinkage::{Binomial, Irls, L1, Problem};
/// let x = Mat::from_fn(4, 1, |i, _| i as f64);
/// let y = [0.0, 0.0, 1.0, 1.0];
/// let fit = Problem::new(&x, Binomial::new(&y), L1::new(0.1))
///     .fit_with(&Irls::new())?;
/// let logits = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct Irls {
    criterion: StoppingCriterion,
    max_iterations: usize,
    max_inner_iterations: usize,
    max_backtracks: usize,
    working_set_policy: WorkingSetPolicy,
}

impl Irls {
    /// Use KKT tolerance `1e-6`, 100 outer iterations, 1,000 inner sweeps per
    /// outer iteration, at most 100 step halvings, and working sets.
    pub fn new() -> Self {
        Self {
            criterion: StoppingCriterion::kkt_violation(1e-6),
            max_iterations: 100,
            max_inner_iterations: 1_000,
            max_backtracks: 100,
            working_set_policy: WorkingSetPolicy::Auto,
        }
    }

    /// Set the finite, strictly positive absolute KKT tolerance.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.criterion = StoppingCriterion::kkt_violation(tolerance);
        self
    }

    /// Select a stopping rule. Only absolute KKT violation is supported.
    pub fn terminate_on(mut self, criterion: StoppingCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the strictly positive maximum number of accepted outer steps.
    pub fn max_iterations(mut self, count: usize) -> Self {
        self.max_iterations = count;
        self
    }

    /// Set the strictly positive sweep limit for each quadratic model.
    /// An inexact inner solve may still supply an accepted descent direction.
    pub fn max_inner_iterations(mut self, count: usize) -> Self {
        self.max_inner_iterations = count;
        self
    }

    /// Set the maximum number of halvings after the initial full-step trial.
    pub fn max_backtracks(mut self, count: usize) -> Self {
        self.max_backtracks = count;
        self
    }

    /// Enable automatic working sets, or disable heuristic restrictions.
    /// The last working-set setter wins.
    pub fn working_sets(mut self, enabled: bool) -> Self {
        self.working_set_policy = WorkingSetPolicy::enabled(enabled);
        self
    }

    /// Select the working-set policy. The last working-set setter wins.
    pub fn working_set_policy(mut self, policy: WorkingSetPolicy) -> Self {
        self.working_set_policy = policy;
        self
    }

    pub(crate) fn validate<E>(&self) -> Result<f64, FitError<E>> {
        self.working_set_policy.validate()?;
        let StoppingCriterion::KktViolation { absolute } = self.criterion else {
            return Err(unsupported(
                "Irls",
                "only absolute KKT stopping is implemented for IRLS",
                "select StoppingCriterion::kkt_violation or Irls::tolerance",
            ));
        };
        if !absolute.is_finite() || absolute <= 0.0 {
            return Err(invalid(
                "KKT tolerance must be finite and strictly positive",
            ));
        }
        if self.max_iterations == 0 || self.max_inner_iterations == 0 {
            return Err(invalid(
                "outer and inner iteration limits must be strictly positive",
            ));
        }
        Ok(absolute)
    }
}

impl Default for Irls {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, D, P> Solver<Problem<&M, D, P>> for Irls
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
    D: DiagonalHessianDatafit,
    P: CoordinatePenalty,
{
    type Fit = IrlsFit;
    type Error = FitError<M::Error>;

    fn solve(&self, problem: &Problem<&M, D, P>) -> Result<IrlsFit, Self::Error> {
        self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut work = Workspace::new(prepared.matrix.nrows(), prepared.matrix.ncols());
        let solution = iterate(
            &prepared.matrix,
            problem.datafit(),
            problem.penalty(),
            self,
            &mut work,
        )?;
        IrlsFit::from_solution(solution, prepared.into_preprocessing())
    }
}

impl<M, D, P> ObservedSolver<Problem<&M, D, P>> for Irls
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
    D: DiagonalHessianDatafit,
    P: CoordinatePenalty,
{
    fn solve_from_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &Problem<&M, D, P>,
        initial: Option<crate::InitialPoint<'_>>,
        observer: &mut O,
    ) -> Result<IrlsFit, Self::Error> {
        self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut work = Workspace::new(prepared.matrix.nrows(), prepared.matrix.ncols());
        if let Some(initial) = initial {
            initial.apply(
                &mut work.coefficients,
                &mut work.intercept,
                problem.datafit().fits_intercept(),
            )?;
        }
        let solution = iterate_observed(
            &prepared.matrix,
            problem.datafit(),
            problem.penalty(),
            self,
            &mut work,
            observer,
        )?;
        IrlsFit::from_solution(solution, prepared.into_preprocessing())
    }
}

pub(crate) fn validate_rows<D: DiagonalHessianDatafit, E>(
    nrows: usize,
    datafit: &D,
) -> Result<(), FitError<E>> {
    if nrows == 0 || datafit.nobs() != nrows {
        return Err(invalid(
            "datafit observation count must match nonempty training rows",
        ));
    }
    Ok(())
}

pub(crate) struct Solution {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub iterations: usize,
    pub coordinate_sweeps: usize,
    pub objective: f64,
    pub kkt_violation: f64,
    pub threshold: f64,
    pub termination: Termination,
    pub working_set: crate::WorkingSetDiagnostics,
}

pub(crate) struct Workspace {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub(crate) eta: Vec<f64>,
    derivative: Vec<f64>,
    weights: Vec<f64>,
    gradient: Vec<f64>,
    inner: Vec<f64>,
    trial: Vec<f64>,
    direction_eta: Vec<f64>,
    trial_eta: Vec<f64>,
    curvatures: Vec<f64>,
    weighted_sums: Vec<f64>,
    working: WorkingSet,
    pub(crate) path: bool,
    residual: WeightedResidual,
}

impl Workspace {
    pub(crate) fn new(n: usize, p: usize) -> Self {
        Self {
            coefficients: vec![0.0; p],
            intercept: 0.0,
            eta: vec![0.0; n],
            derivative: vec![0.0; n],
            weights: vec![0.0; n],
            gradient: vec![0.0; p],
            inner: vec![0.0; p],
            trial: vec![0.0; p],
            direction_eta: vec![0.0; n],
            trial_eta: vec![0.0; n],
            curvatures: vec![0.0; p],
            weighted_sums: vec![0.0; p],
            working: WorkingSet::new(p),
            path: false,
            residual: WeightedResidual {
                base: vec![0.0; n],
                sum: 0.0,
                offset: 0.0,
            },
        }
    }

    fn evaluate<M, D, P, E>(
        &mut self,
        matrix: &LazyMatrix<M>,
        datafit: &D,
        penalty: &P,
        iteration: usize,
    ) -> Result<(f64, f64), FitError<E>>
    where
        M: RawColumns<f64>,
        D: DiagonalHessianDatafit,
        P: CoordinatePenalty,
    {
        // Reconstruct rather than accumulating predictor error across outer steps.
        predictor(
            matrix,
            self.coefficients.iter().copied(),
            self.intercept,
            &mut self.eta,
            iteration,
        )?;
        let loss = datafit
            .value(&self.eta)
            .map_err(|e| FitError::from_component(e, iteration))?;
        let penalty_value = penalty
            .value(&self.coefficients)
            .map_err(|e| FitError::from_component(e, iteration))?;
        let objective = finite(
            loss + penalty_value,
            "computing the IRLS objective",
            iteration,
        )?;
        datafit
            .gradient(&self.eta, &mut self.derivative)
            .map_err(|e| FitError::from_component(e, iteration))?;
        for &value in &self.derivative {
            finite(value, "checking the predictor gradient", iteration)?;
        }
        let sum = finite(
            self.derivative.iter().sum(),
            "summing the predictor gradient",
            iteration,
        )?;
        let mut violation = if datafit.fits_intercept() {
            sum.abs()
        } else {
            0.0
        };
        for (j, gradient) in self.gradient.iter_mut().enumerate() {
            *gradient = finite(
                matrix.column(j).dot_with_sum(&self.derivative, sum),
                "computing the coefficient gradient",
                iteration,
            )?;
            self.working.scores[j] =
                checked_violation(penalty, j, self.coefficients[j], *gradient, iteration)?;
            violation = violation.max(self.working.scores[j]);
        }
        self.working.diagnostics.full_checks += 1;
        Ok((objective, violation))
    }

    fn quadratic<M, D, P, E, O>(
        &mut self,
        matrix: &LazyMatrix<M>,
        datafit: &D,
        penalty: &P,
        options: &Irls,
        control: (f64, usize),
        observer: &mut O,
    ) -> Result<(f64, usize), FitError<E>>
    where
        M: RawColumns<f64>,
        D: DiagonalHessianDatafit,
        P: CoordinatePenalty,
        O: SolverObserver + ?Sized,
    {
        let (target, iteration) = control;
        datafit
            .hessian_diagonal(&self.eta, &mut self.weights)
            .map_err(|e| FitError::from_component(e, iteration))?;
        let floor = 1e-12 / self.eta.len() as f64;
        for weight in &mut self.weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(invalid(
                    "predictor Hessian entries must be finite and nonnegative",
                ));
            }
            *weight = weight.max(floor);
        }
        let weight_sum = finite(self.weights.iter().sum(), "summing IRLS weights", iteration)?;
        for j in 0..self.coefficients.len() {
            let column = matrix.column(j);
            self.curvatures[j] = finite(
                column.weighted_norm_squared(&self.weights),
                "computing weighted column curvature",
                iteration,
            )?;
            if self.curvatures[j] < 0.0 {
                return Err(invalid("weighted column curvature must be nonnegative"));
            }
            self.weighted_sums[j] = finite(
                column.dot_with_sum(&self.weights, weight_sum),
                "computing weighted column sums",
                iteration,
            )?;
        }
        self.inner.copy_from_slice(&self.coefficients);
        let mut intercept = self.intercept;
        for (r, &g) in self.residual.base.iter_mut().zip(&self.derivative) {
            *r = -g;
        }
        self.residual.sum = finite(
            self.residual.base.iter().sum(),
            "summing quadratic residuals",
            iteration,
        )?;
        self.residual.offset = 0.0;
        self.working.begin_model(&self.inner);
        let mut maximum = self.check_quadratic(matrix, penalty, iteration, true)?;
        if datafit.fits_intercept() {
            maximum = maximum.max(self.residual.sum.abs());
        }
        self.working.select(&self.inner, &[], false, false, target);
        let mut inner_target = target.max(0.1 * maximum);
        let mut moved = false;
        let mut since_full = 0;
        for sweep in 1..=options.max_inner_iterations {
            since_full += 1;
            for &j in &self.working.active {
                self.working.diagnostics.feature_visits += 1;
                // Positive row weights make a zero norm a logically zero
                // column. Its exact correlation is zero, even if subtracting
                // raw centered products leaves a small rounding remainder.
                let correlation = if self.curvatures[j] == 0.0 {
                    0.0
                } else {
                    self.residual
                        .dot(matrix, j, self.weighted_sums[j], iteration)?
                };
                let linear = finite(
                    correlation + self.curvatures[j] * self.inner[j],
                    "computing a quadratic coordinate update",
                    iteration,
                )?;
                let updated = penalty
                    .minimize_coordinate(j, linear, self.curvatures[j])
                    .map_err(|e| FitError::from_component(e, iteration))?;
                finite(updated, "checking a coordinate minimizer", iteration)?;
                let delta = finite(
                    updated - self.inner[j],
                    "computing a coordinate change",
                    iteration,
                )?;
                if delta != 0.0 {
                    self.residual
                        .subtract(matrix, j, delta, &self.weights, iteration)?;
                    self.inner[j] = updated;
                    moved = true;
                    notify_inner(
                        observer,
                        &self.inner,
                        intercept,
                        iteration,
                        sweep,
                        StepKind::Coordinate { index: j },
                    );
                }
            }
            if datafit.fits_intercept() {
                let delta = finite(
                    self.residual.sum / weight_sum + self.residual.offset,
                    "updating the weighted intercept",
                    iteration,
                )?;
                intercept = finite(intercept + delta, "updating the IRLS intercept", iteration)?;
                self.residual.offset = finite(
                    self.residual.offset - delta,
                    "updating the weighted residual offset",
                    iteration,
                )?;
                if delta != 0.0 {
                    moved = true;
                    notify_inner(
                        observer,
                        &self.inner,
                        intercept,
                        iteration,
                        sweep,
                        StepKind::Intercept,
                    );
                }
            }
            self.working.record(&self.inner);
            let disabled = self.working.diagnostics.policy == WorkingSetPolicy::Disabled;
            let local = if disabled {
                f64::INFINITY
            } else {
                let mut local = self.check_quadratic(matrix, penalty, iteration, false)?;
                if datafit.fits_intercept() {
                    local =
                        local.max((self.residual.sum + self.residual.offset * weight_sum).abs());
                }
                local
            };
            let discover =
                sweep == 1 && self.working.diagnostics.policy == WorkingSetPolicy::EverActive;
            if local <= inner_target
                || !moved
                || since_full == 10
                || disabled
                || discover
                || sweep == options.max_inner_iterations
            {
                // Only a reconstructed model can accept a quadratic solution.
                predictor(
                    matrix,
                    self.inner
                        .iter()
                        .zip(&self.coefficients)
                        .map(|(a, b)| a - b),
                    intercept - self.intercept,
                    &mut self.direction_eta,
                    iteration,
                )?;
                for ((r, &g), (&h, &d)) in self
                    .residual
                    .base
                    .iter_mut()
                    .zip(&self.derivative)
                    .zip(self.weights.iter().zip(&self.direction_eta))
                {
                    *r = finite(-g - h * d, "refreshing quadratic residuals", iteration)?;
                }
                self.residual.sum = finite(
                    self.residual.base.iter().sum(),
                    "refreshing the quadratic residual sum",
                    iteration,
                )?;
                self.residual.offset = 0.0;
                let mut violation = self.check_quadratic(matrix, penalty, iteration, true)?;
                if datafit.fits_intercept() {
                    violation = violation.max(self.residual.sum.abs());
                }
                self.working.select(
                    &self.inner,
                    &[],
                    local <= inner_target || !moved,
                    !moved,
                    target,
                );
                if violation <= target {
                    return Ok((intercept, sweep));
                }
                inner_target = target.max(0.1 * violation);
                since_full = 0;
                moved = false;
            }
        }
        Ok((intercept, options.max_inner_iterations))
    }

    fn check_quadratic<M, P, E>(
        &mut self,
        matrix: &LazyMatrix<M>,
        penalty: &P,
        iteration: usize,
        full: bool,
    ) -> Result<f64, FitError<E>>
    where
        M: RawColumns<f64>,
        P: CoordinatePenalty,
    {
        let mut maximum: f64 = 0.0;
        for j in 0..self.inner.len() {
            if !full && !self.working.selected[j] {
                continue;
            }
            let gradient = if self.curvatures[j] == 0.0 {
                0.0
            } else {
                -self
                    .residual
                    .dot(matrix, j, self.weighted_sums[j], iteration)?
            };
            let violation = checked_violation(penalty, j, self.inner[j], gradient, iteration)?;
            self.working.scores[j] = violation;
            maximum = maximum.max(violation);
        }
        if full {
            self.working.diagnostics.full_checks += 1;
        } else {
            self.working.diagnostics.restricted_checks += 1;
        }
        Ok(maximum)
    }
}

// The logical residual is base[i] + weight[i] * offset. This represents
// centered-column changes without touching every implicit zero during an update.
struct WeightedResidual {
    base: Vec<f64>,
    sum: f64,
    offset: f64,
}

impl WeightedResidual {
    fn dot<M: RawColumns<f64>, E>(
        &self,
        matrix: &LazyMatrix<M>,
        j: usize,
        weighted_sum: f64,
        iteration: usize,
    ) -> Result<f64, FitError<E>> {
        Ok(finite(
            matrix.column(j).dot_with_sum(&self.base, self.sum) + self.offset * weighted_sum,
            "computing a quadratic residual correlation",
            iteration,
        )?)
    }

    fn subtract<M: RawColumns<f64>, E>(
        &mut self,
        matrix: &LazyMatrix<M>,
        j: usize,
        delta: f64,
        weights: &[f64],
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        let column = matrix.column(j);
        let multiplier = finite(
            delta / column.scale(),
            "scaling a weighted coordinate change",
            iteration,
        )?;
        let mut change_sum = 0.0;
        let mut valid = true;
        column.raw().for_each_stored(|i, value| {
            let change = multiplier * value * weights[i];
            self.base[i] -= change;
            change_sum += change;
            valid &= self.base[i].is_finite();
        });
        if !valid {
            return Err(FitError::NumericalFailure {
                operation: "updating weighted residuals",
                iteration,
            });
        }
        self.sum = finite(
            self.sum - change_sum,
            "updating the weighted residual sum",
            iteration,
        )?;
        self.offset = finite(
            self.offset + multiplier * column.center(),
            "updating the weighted residual offset",
            iteration,
        )?;
        Ok(())
    }
}

fn checked_violation<P: CoordinatePenalty, E>(
    penalty: &P,
    j: usize,
    value: f64,
    gradient: f64,
    iteration: usize,
) -> Result<f64, FitError<E>> {
    let violation = penalty
        .coordinate_violation(j, value, gradient)
        .map_err(|e| FitError::from_component(e, iteration))?;
    finite(violation, "checking coordinate KKT violation", iteration)?;
    if violation < 0.0 {
        return Err(invalid("coordinate KKT violations must be nonnegative"));
    }
    Ok(violation)
}

pub(crate) fn predictor<M: RawColumns<f64>, E>(
    matrix: &LazyMatrix<M>,
    coefficients: impl Iterator<Item = f64>,
    intercept: f64,
    output: &mut [f64],
    iteration: usize,
) -> Result<(), FitError<E>> {
    output.fill(0.0);
    let mut offset = intercept;
    for (j, coefficient) in coefficients.enumerate() {
        if coefficient == 0.0 {
            continue;
        }
        let column = matrix.column(j);
        let multiplier = finite(
            coefficient / column.scale(),
            "scaling the IRLS predictor",
            iteration,
        )?;
        column
            .raw()
            .for_each_stored(|i, value| output[i] += multiplier * value);
        offset = finite(
            offset - multiplier * column.center(),
            "centering the IRLS predictor",
            iteration,
        )?;
    }
    for value in output {
        *value = finite(*value + offset, "computing the IRLS predictor", iteration)?;
    }
    Ok(())
}

pub(crate) fn iterate<M, D, P>(
    matrix: &LazyMatrix<M>,
    datafit: &D,
    penalty: &P,
    options: &Irls,
    work: &mut Workspace,
) -> Result<Solution, FitError<M::Error>>
where
    M: RawColumns<f64> + MatrixErrorType,
    D: DiagonalHessianDatafit,
    P: CoordinatePenalty,
{
    iterate_observed(matrix, datafit, penalty, options, work, &mut NoObserver)
}

pub(crate) fn iterate_observed<M, D, P, T>(
    matrix: &LazyMatrix<M>,
    datafit: &D,
    penalty: &P,
    options: &Irls,
    work: &mut Workspace,
    observer: &mut T,
) -> Result<Solution, FitError<M::Error>>
where
    T: SolverObserver + ?Sized,
    M: RawColumns<f64> + MatrixErrorType,
    D: DiagonalHessianDatafit,
    P: CoordinatePenalty,
{
    let threshold = options.validate()?;
    work.working
        .begin(options.working_set_policy, work.path, &work.coefficients);
    penalty
        .validate_input(&work.coefficients)
        .map_err(|e| FitError::from_component(e, 0))?;
    let mut iterations = 0;
    let mut coordinate_sweeps = 0;
    notify(
        observer,
        &work.coefficients,
        work.intercept,
        0,
        StepKind::Initial,
    );
    loop {
        let (objective, violation) = work.evaluate(matrix, datafit, penalty, iterations)?;
        if violation <= threshold || iterations == options.max_iterations {
            work.working
                .select(&work.coefficients, &[], false, false, threshold);
            notify(
                observer,
                &work.coefficients,
                work.intercept,
                iterations,
                StepKind::Final(if violation <= threshold {
                    Termination::Converged
                } else {
                    Termination::IterationLimit
                }),
            );
            return Ok(Solution {
                coefficients: work.coefficients.clone(),
                intercept: work.intercept,
                iterations,
                coordinate_sweeps,
                working_set: work.working.diagnostics.clone(),
                objective,
                kkt_violation: violation,
                threshold,
                termination: if violation <= threshold {
                    Termination::Converged
                } else {
                    Termination::IterationLimit
                },
            });
        }
        let next = iterations + 1;
        let target = 0.1 * violation.max(threshold);
        let (inner_intercept, sweeps) =
            work.quadratic(matrix, datafit, penalty, options, (target, next), observer)?;
        coordinate_sweeps += sweeps;
        let intercept_delta = inner_intercept - work.intercept;
        predictor(
            matrix,
            work.inner
                .iter()
                .zip(&work.coefficients)
                .map(|(a, b)| a - b),
            intercept_delta,
            &mut work.direction_eta,
            next,
        )?;
        // Coordinate minimization decreases the quadratic model, so
        // g'd + R(beta+d) - R(beta) <= -d'Hd/2. This upper bound
        // avoids subtracting nearly equal penalty values near convergence.
        let slope = finite(
            -0.5 * work
                .weights
                .iter()
                .zip(&work.direction_eta)
                .map(|(h, d)| (h * d) * d)
                .sum::<f64>(),
            "computing the IRLS directional decrease",
            next,
        )?;
        if inner_intercept == work.intercept && work.inner == work.coefficients {
            return Err(FitError::LineSearchFailure {
                iteration: next,
                step: 0.0,
            });
        }
        let mut step = 1.0;
        let mut accepted = false;
        for backtrack in 0..=options.max_backtracks {
            for ((trial, &current), &candidate) in work
                .trial
                .iter_mut()
                .zip(&work.coefficients)
                .zip(&work.inner)
            {
                *trial = current + step * (candidate - current);
            }
            let trial_intercept = work.intercept + step * intercept_delta;
            for ((trial, &eta), &delta) in work
                .trial_eta
                .iter_mut()
                .zip(&work.eta)
                .zip(&work.direction_eta)
            {
                *trial = eta + step * delta;
            }
            let valid = trial_intercept.is_finite()
                && work
                    .trial
                    .iter()
                    .chain(&work.trial_eta)
                    .all(|v| v.is_finite());
            if valid {
                let value = (|| {
                    let loss = datafit
                        .value(&work.trial_eta)
                        .map_err(|e| FitError::from_component(e, next))?;
                    let regularization = penalty
                        .value(&work.trial)
                        .map_err(|e| FitError::from_component(e, next))?;
                    Ok::<_, FitError<M::Error>>(finite(
                        loss + regularization,
                        "computing a trial objective",
                        next,
                    )?)
                })();
                match value {
                    Ok(value) => {
                        let slack = 16.0 * f64::EPSILON * objective.abs().max(value.abs());
                        if value <= objective + 1e-4 * step * slope + slack {
                            work.coefficients.copy_from_slice(&work.trial);
                            work.intercept = trial_intercept;
                            accepted = true;
                            break;
                        }
                    }
                    Err(FitError::NumericalFailure { .. }) => {}
                    Err(error) => return Err(error),
                }
            }
            if backtrack < options.max_backtracks {
                step *= 0.5;
            }
        }
        if !accepted {
            return Err(FitError::LineSearchFailure {
                iteration: next,
                step,
            });
        }
        work.working.record(&work.coefficients);
        iterations = next;
        notify(
            observer,
            &work.coefficients,
            work.intercept,
            iterations,
            StepKind::Irls,
        );
    }
}
