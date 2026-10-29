//! Logistic globalization around reusable weighted hybrid subproblems.

use super::*;
use crate::{Binomial, Datafit, DiagonalHessianDatafit};

impl<M, V, P> Solver<Problem<&M, Binomial<'_>, P>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
    P: HybridPenalty,
{
    type Fit = HybridFit;
    type Error = FitError<M::Error>;

    fn solve(&self, problem: &Problem<&M, Binomial<'_>, P>) -> Result<HybridFit, Self::Error> {
        let threshold = self.validate()?;
        if self.max_outer_iterations == 0 || self.max_inner_iterations == 0 {
            return Err(invalid(
                "logistic outer and inner iteration limits must be positive",
            ));
        }
        let n = problem.design().nrows();
        let p = problem.design().ncols();
        let datafit = *problem.datafit();
        if n == 0 || datafit.nobs() != n {
            return Err(invalid(
                "datafit observation count must match nonempty training rows",
            ));
        }
        let mut inner = Workspace::<P>::new(n, p);
        datafit
            .value(&inner.residual)
            .map_err(|e| FitError::from_component(e, 0))?;
        problem
            .penalty()
            .validate_input(&inner.coefficients)
            .map_err(|e| FitError::from_component(e, 0))?;
        let prepared = problem.prepare(datafit.fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        let solution = iterate(
            &mut design,
            datafit,
            problem.penalty(),
            self,
            threshold,
            &mut inner,
        )?;
        HybridFit::from_solution(solution, design.into_preprocessing())
    }
}

impl<M, V, P> ObservedSolver<Problem<&M, Binomial<'_>, P>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
    P: HybridPenalty,
{
    fn solve_from_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &Problem<&M, Binomial<'_>, P>,
        initial: Option<crate::InitialPoint<'_>>,
        observer: &mut O,
    ) -> Result<HybridFit, Self::Error> {
        let threshold = self.validate()?;
        if self.max_outer_iterations == 0 || self.max_inner_iterations == 0 {
            return Err(invalid(
                "logistic outer and inner iteration limits must be positive",
            ));
        }
        let n = problem.design().nrows();
        let p = problem.design().ncols();
        let datafit = *problem.datafit();
        if n == 0 || datafit.nobs() != n {
            return Err(invalid(
                "datafit observation count must match nonempty training rows",
            ));
        }
        let mut inner = Workspace::<P>::new(n, p);
        datafit
            .value(&inner.residual)
            .map_err(|e| FitError::from_component(e, 0))?;
        problem
            .penalty()
            .validate_input(&inner.coefficients)
            .map_err(|e| FitError::from_component(e, 0))?;
        let prepared = problem.prepare(datafit.fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        if let Some(initial) = initial {
            initial.apply(
                &mut inner.coefficients,
                &mut inner.intercept,
                datafit.fits_intercept(),
            )?;
        }
        let solution = iterate_observed(
            &mut design,
            datafit,
            problem.penalty(),
            self,
            threshold,
            &mut inner,
            observer,
        )?;
        HybridFit::from_solution(solution, design.into_preprocessing())
    }
}

struct Outer {
    coefficients: Vec<f64>,
    intercept: f64,
    eta: Vec<f64>,
    derivative: Vec<f64>,
    weights: Vec<f64>,
    direction_eta: Vec<f64>,
    trial_eta: Vec<f64>,
    trial: Vec<f64>,
}

fn iterate<M, V, W, P>(
    design: &mut NativeDesign<M, V>,
    datafit: Binomial<'_>,
    penalty: &P,
    options: &Hybrid<W>,
    threshold: f64,
    inner: &mut Workspace<P>,
) -> Result<Solution, FitError<M::Error>>
where
    M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
    P: HybridPenalty,
{
    iterate_observed(
        design,
        datafit,
        penalty,
        options,
        threshold,
        inner,
        &mut NoObserver,
    )
}

fn iterate_observed<M, V, W, P, T>(
    design: &mut NativeDesign<M, V>,
    datafit: Binomial<'_>,
    penalty: &P,
    options: &Hybrid<W>,
    threshold: f64,
    inner: &mut Workspace<P>,
    observer: &mut T,
) -> Result<Solution, FitError<M::Error>>
where
    T: SolverObserver + ?Sized,
    M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
    P: HybridPenalty,
{
    inner
        .working
        .begin(options.working_set_policy, false, &inner.coefficients);
    inner.screening.reset();
    inner.state.set_screened(&inner.screening.zeroed);
    inner.pending.set_screened(&inner.screening.zeroed);
    let n = design.nrows();
    let mut outer = Outer {
        coefficients: inner.coefficients.clone(),
        intercept: inner.intercept,
        eta: vec![0.0; n],
        derivative: vec![0.0; n],
        weights: vec![0.0; n],
        direction_eta: vec![0.0; n],
        trial_eta: vec![0.0; n],
        trial: vec![0.0; design.ncols()],
    };
    let mut inner_options = Hybrid::<()>::new().step_policy(HybridStep::FixedBound);
    let curvature = finite(
        0.25 * initial_curvature(
            design,
            datafit.fits_intercept(),
            &Weights::default(),
            &inner_options,
        )?,
        "bounding logistic curvature",
        0,
    )?;
    finite(1.0 / curvature, "computing the logistic diagnostic step", 0)?;
    inner.reference_curvature = Some(curvature);
    inner_options.cluster_sweeps = options.cluster_sweeps;
    inner_options.check_every = options.check_every;
    inner_options.step = options.step;
    inner_options.screening = false;
    inner_options.working_set_policy = options.working_set_policy.resolve(false);
    let mut total = Progress {
        iterations: 0,
        objective: 0.0,
        mapping: 0.0,
        curvature,
        termination: Termination::IterationLimit,
        pg_steps: 0,
        coordinate_sweeps: 0,
        diagnostic_checks: 0,
        backtracks: 0,
    };
    let (mut outer_iterations, mut outer_backtracks) = (0, 0);
    notify(
        observer,
        &outer.coefficients,
        outer.intercept,
        0,
        StepKind::Initial,
    );
    loop {
        inner.coefficients.copy_from_slice(&outer.coefficients);
        inner.intercept = outer.intercept;
        inner
            .screening
            .forward(design, &outer.coefficients, &mut outer.eta)?;
        for eta in &mut outer.eta {
            *eta = finite(
                *eta + outer.intercept,
                "refreshing logistic predictors",
                outer_iterations,
            )?;
        }
        let regularization = penalty
            .value(&outer.coefficients)
            .map_err(|e| FitError::from_component(e, outer_iterations))?;
        let objective = finite(
            datafit
                .value(&outer.eta)
                .map_err(|e| FitError::from_component(e, outer_iterations))?
                + regularization,
            "computing the logistic hybrid objective",
            outer_iterations,
        )?;
        datafit
            .gradient(&outer.eta, &mut outer.derivative)
            .map_err(|e| FitError::from_component(e, outer_iterations))?;
        design.transpose(&outer.derivative, &mut inner.gradient)?;
        let intercept_gradient = if datafit.fits_intercept() {
            finite(
                outer.derivative.iter().sum(),
                "computing the logistic intercept gradient",
                outer_iterations,
            )?
        } else {
            0.0
        };
        inner.trial::<M::Error>(penalty, curvature, intercept_gradient, outer_iterations)?;
        let mut mapping = intercept_gradient.abs();
        for (j, (&b, &z)) in outer.coefficients.iter().zip(&inner.candidate).enumerate() {
            inner.working.scores[j] = finite(
                (b - z) * curvature,
                "checking logistic working-set scores",
                outer_iterations,
            )?
            .abs();
            mapping = mapping.max(
                finite(
                    (b - z) * curvature,
                    "checking original-loss logistic stationarity",
                    outer_iterations,
                )?
                .abs(),
            );
        }
        inner.working.diagnostics.full_checks += 1;
        if options.screening
            && inner.screening.check(
                design,
                &datafit,
                penalty,
                &outer.coefficients,
                &outer.eta,
                objective,
                outer_iterations,
                mapping > threshold
                    && outer_iterations < options.max_outer_iterations
                    && total.iterations < options.max_iterations,
            )?
        {
            inner.state.set_screened(&inner.screening.zeroed);
            inner.pending.set_screened(&inner.screening.zeroed);
        }
        if mapping <= threshold
            || outer_iterations == options.max_outer_iterations
            || total.iterations == options.max_iterations
        {
            inner.working.select(
                &inner.coefficients,
                &inner.screening.zeroed,
                false,
                false,
                threshold,
            );
            inner.pending.set_screened(&inner.working.zeroed);
            for (r, &g) in inner.residual.iter_mut().zip(&outer.derivative) {
                *r = -(n as f64) * g;
            }
            inner.pending.rebuild(
                design.matrix(),
                &inner.coefficients,
                &inner.residual,
                penalty,
                total.iterations,
            )?;
            total.objective = objective;
            total.mapping = mapping;
            total.termination = if mapping <= threshold {
                Termination::Converged
            } else {
                Termination::IterationLimit
            };
            let mut solution = total.finish(inner, options, threshold);
            solution.outer_iterations = outer_iterations;
            solution.outer_backtracks = outer_backtracks;
            notify(
                observer,
                &outer.coefficients,
                outer.intercept,
                solution.proximal.iterations,
                StepKind::Final(solution.proximal.termination),
            );
            return Ok(solution);
        }
        datafit
            .hessian_diagonal(&outer.eta, &mut outer.weights)
            .map_err(|e| FitError::from_component(e, outer_iterations))?;
        // Scaled weights preserve the existing averaged Gaussian kernels. Only
        // the surrogate is floored; diagnostics always use the original loss.
        for weight in &mut outer.weights {
            *weight = (*weight * n as f64).max(1e-12);
        }
        let model = Model::Newton {
            anchor: &outer.eta,
            gradient: &outer.derivative,
            weights: &outer.weights,
            fit_intercept: datafit.fits_intercept(),
        };
        inner_options.max_iterations = options
            .max_inner_iterations
            .min(options.max_iterations - total.iterations);
        let report = iterate_quadratic_observed(
            design,
            model,
            penalty,
            &inner_options,
            0.1 * mapping.max(threshold),
            inner,
            &mut InnerObserver {
                observer,
                outer_iteration: outer_iterations + 1,
            },
        )?;
        total.iterations += report.iterations;
        total.pg_steps += report.pg_steps;
        total.coordinate_sweeps += report.coordinate_sweeps;
        total.diagnostic_checks += report.diagnostic_checks;
        total.backtracks += report.backtracks;
        let intercept_delta = finite(
            inner.intercept - outer.intercept,
            "forming an IRLS intercept direction",
            outer_iterations,
        )?;
        for ((d, &next), &base) in inner
            .direction
            .iter_mut()
            .zip(&inner.coefficients)
            .zip(&outer.coefficients)
        {
            *d = finite(
                next - base,
                "forming a hybrid IRLS direction",
                outer_iterations,
            )?;
        }
        if intercept_delta == 0.0 && inner.direction.iter().all(|&d| d == 0.0) {
            return Err(FitError::LineSearchFailure {
                iteration: outer_iterations,
                step: 0.0,
            });
        }
        inner
            .screening
            .forward(design, &inner.direction, &mut outer.direction_eta)?;
        let (mut linear, mut quadratic) = (0.0, 0.0);
        for ((d, &g), &w) in outer
            .direction_eta
            .iter_mut()
            .zip(&outer.derivative)
            .zip(&outer.weights)
        {
            *d = finite(
                *d + intercept_delta,
                "forming an IRLS predictor direction",
                outer_iterations,
            )?;
            linear += g * *d;
            quadratic += (w * *d) * (*d / n as f64);
        }
        let quadratic = finite(
            quadratic,
            "computing IRLS directional curvature",
            outer_iterations,
        )?;
        let next_penalty = penalty
            .value(&inner.coefficients)
            .map_err(|e| FitError::from_component(e, outer_iterations))?;
        let change = finite(
            linear + 0.5 * quadratic + (next_penalty - regularization),
            "checking quadratic descent",
            outer_iterations,
        )?;
        let slack = 64.0
            * f64::EPSILON
            * (linear.abs() + quadratic + next_penalty.abs() + regularization.abs());
        if change > slack {
            return Err(FitError::LineSearchFailure {
                iteration: outer_iterations,
                step: 0.0,
            });
        }
        // Model descent bounds the composite slope by -d'Hd/2 without
        // subtracting nearly equal penalty values during globalization.
        let slope = -0.5 * quadratic;
        let mut step = 1.0;
        let mut accepted = false;
        for backtrack in 0..=options.max_outer_backtracks {
            for ((trial, &base), &d) in outer
                .trial
                .iter_mut()
                .zip(&outer.coefficients)
                .zip(&inner.direction)
            {
                *trial = base + step * d;
            }
            for ((trial, &base), &d) in outer
                .trial_eta
                .iter_mut()
                .zip(&outer.eta)
                .zip(&outer.direction_eta)
            {
                *trial = base + step * d;
            }
            let intercept = outer.intercept + step * intercept_delta;
            if intercept.is_finite()
                && outer
                    .trial
                    .iter()
                    .chain(&outer.trial_eta)
                    .all(|v| v.is_finite())
            {
                let trial_objective = (|| {
                    let loss = datafit
                        .value(&outer.trial_eta)
                        .map_err(|e| FitError::from_component(e, outer_iterations))?;
                    let value = penalty
                        .value(&outer.trial)
                        .map_err(|e| FitError::from_component(e, outer_iterations))?;
                    Ok::<_, FitError<M::Error>>(finite(
                        loss + value,
                        "computing a logistic trial objective",
                        outer_iterations,
                    )?)
                })();
                match trial_objective {
                    Ok(value) => {
                        let slack = 16.0 * f64::EPSILON * objective.abs().max(value.abs());
                        if value <= objective + 1e-4 * step * slope + slack {
                            outer.coefficients.copy_from_slice(&outer.trial);
                            outer.intercept = intercept;
                            accepted = true;
                            break;
                        }
                    }
                    Err(FitError::NumericalFailure { .. }) => {}
                    Err(error) => return Err(error),
                }
            }
            if backtrack < options.max_outer_backtracks {
                step *= 0.5;
                outer_backtracks += 1;
            }
        }
        if !accepted {
            return Err(FitError::LineSearchFailure {
                iteration: outer_iterations,
                step,
            });
        }
        outer_iterations += 1;
        notify(
            observer,
            &outer.coefficients,
            outer.intercept,
            total.iterations,
            StepKind::Irls,
        );
    }
}
