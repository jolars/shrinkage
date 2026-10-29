//! Backtracking proximal gradient over concrete or object-safe design oracles.

mod design;
mod multivariate;
pub(crate) use design::NativeDesign;
pub use design::ProximalDesign;

use super::observer::{NoObserver, notify};
use super::{ObservedSolver, SolverObserver, StepKind};
use std::marker::PhantomData;

use lazymatrix::{
    ColumnStats, DotSlice, ElemDivAssign, MatTransposeVecInto, MatVecInto, RawColumns,
    ScaledSubSlice, SubScalarAssign, SumEntries, VectorOwned, VectorView, VectorViewMut,
};

use crate::error::{finite, invalid, unsupported};
use crate::{
    FitError, Problem, ProximalFit, ProximalPenalty, SmoothDatafit, Solver, StoppingCriterion,
    Termination,
};

/// Owned CPU storage for a backend's product vectors.
///
/// Matrix products and normalization remain in LazyMatrix. This capability
/// supplies reusable buffers and bridges native vectors to the slice-based
/// predictor and proximal capabilities. It does not represent device storage.
///
/// LazyMatrix vectors with `VectorOwned<Owned = Self>`, mutable indexed access,
/// and the normalization operations below implement this trait automatically.
/// Indexed access may be strided; it does not require contiguous storage.
pub trait ProximalVector:
    Clone
    + ElemDivAssign<f64>
    + DotSlice<f64>
    + SubScalarAssign<f64>
    + SumEntries<f64>
    + ScaledSubSlice<f64>
{
    /// Allocate a zero vector of the requested length.
    fn zeros(length: usize) -> Self;
    /// Read an entry in constant time.
    fn get(&self, index: usize) -> f64;
    /// Write an entry in constant time.
    fn set(&mut self, index: usize, value: f64);
}

impl<V> ProximalVector for V
where
    V: Clone
        + VectorOwned<f64, Owned = V>
        + VectorViewMut<f64>
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    fn zeros(length: usize) -> Self {
        V::owned_from_fn(length, |_| 0.0)
    }
    fn get(&self, index: usize) -> f64 {
        VectorView::get(self, index)
    }
    fn set(&mut self, index: usize, value: f64) {
        VectorViewMut::set(self, index, value);
    }
}

/// Proximal gradient with backtracking for convex smooth predictor losses and
/// complete convex proximal penalties.
///
/// `V` is the design backend's owned vector: `faer::Col<f64>`,
/// `nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, or `Vec<f64>` for sprs.
/// No normalized design or Gram matrix is formed. LazyMatrix centers dense
/// products before accumulation. Its generic scaled forward product still
/// clones the input when the backend does not specialize normalization.
/// [`crate::MultivariateGaussian`] with [`crate::AxisPenalty`] fits joint
/// matrix coefficients and response-specific unpenalized intercepts through
/// the same iteration, returning [`crate::MultivariateFit`].
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::{Col, Mat};
/// use shrinkage::{Gaussian, Problem, ProximalGradient, Ridge};
/// let x = Mat::from_fn(3, 1, |i, _| i as f64);
/// let y = [1.0, 3.0, 5.0];
/// let fit = Problem::new(&x, Gaussian::new(&y), Ridge::new(0.1))
///     .fit_with(&ProximalGradient::<Col<f64>>::new())?;
/// let predictions = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ProximalGradient<V = Vec<f64>> {
    criterion: StoppingCriterion,
    max_iterations: usize,
    initial_step: f64,
    max_backtracks: usize,
    screening: bool,
    vector: PhantomData<V>,
}

impl<V> ProximalGradient<V> {
    /// Use an absolute mapping tolerance of `1e-6`, 10,000 iterations, initial
    /// step one, and at most 100 step halvings per iteration.
    pub fn new() -> Self {
        Self {
            criterion: StoppingCriterion::proximal_gradient_mapping(1e-6),
            max_iterations: 10_000,
            initial_step: 1.0,
            max_backtracks: 100,
            screening: true,
            vector: PhantomData,
        }
    }

    /// Set the absolute infinity-norm proximal-gradient mapping tolerance.
    pub fn tolerance(mut self, absolute: f64) -> Self {
        self.criterion = StoppingCriterion::proximal_gradient_mapping(absolute);
        self
    }

    /// Select a criterion. This solver supports proximal-gradient mapping only.
    pub fn terminate_on(mut self, criterion: StoppingCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the maximum number of accepted proximal updates.
    pub fn max_iterations(mut self, count: usize) -> Self {
        self.max_iterations = count;
        self
    }

    /// Set a finite positive initial step, halved until the smooth loss obeys
    /// its quadratic upper bound. Accepted steps are retained across iterations.
    pub fn initial_step(mut self, step: f64) -> Self {
        self.initial_step = step;
        self
    }

    /// Set the maximum number of step halvings for each line search.
    pub fn max_backtracks(mut self, count: usize) -> Self {
        self.max_backtracks = count;
        self
    }

    /// Enable dynamic Gap Safe zero screening when every component supports it.
    ///
    /// Enabled by default. Certificates use the original loss and penalty;
    /// unsupported compositions retain ordinary proximal-gradient fitting.
    pub fn screening(mut self, enabled: bool) -> Self {
        self.screening = enabled;
        self
    }

    /// Fit penalties in the supplied order, preparing the design once.
    ///
    /// The problem's penalty is replaced by each path entry. Native product
    /// buffers and iteration storage are reused; each point starts from the
    /// preceding optimization-scale coefficients and intercept. Each point
    /// resets its step and iteration budget. Results own independent parameters
    /// and preprocessing. An empty path returns no fits.
    ///
    /// # Errors
    /// Returns validation, component, line-search, or backend errors without
    /// returning a partial path. Iteration-limit results remain in the path and
    /// provide the next warm start.
    pub fn solve_path<M, D, P, Q>(
        &self,
        problem: &Problem<&M, D, P>,
        penalties: &[Q],
    ) -> Result<Vec<ProximalFit>, FitError<M::Error>>
    where
        M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
        D: SmoothDatafit,
        Q: ProximalPenalty,
        V: ProximalVector,
    {
        let threshold = self.validate()?;
        if problem.datafit().nobs() != problem.design().nrows() {
            return Err(invalid(
                "datafit observation count must match training rows",
            ));
        }
        if penalties.is_empty() {
            return Ok(Vec::new());
        }
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        let solutions = iterate_path(&mut design, problem.datafit(), penalties, self, threshold)?;
        let preprocessing = design.into_preprocessing();
        solutions
            .into_iter()
            .map(|solution| ProximalFit::from_solution(solution, preprocessing.clone()))
            .collect()
    }

    pub(crate) fn validate<E>(&self) -> Result<f64, FitError<E>> {
        let StoppingCriterion::ProximalGradientMapping { absolute } = self.criterion else {
            return Err(unsupported(
                "ProximalGradient",
                format!(
                    "stopping criterion {:?} is not implemented for proximal gradient",
                    self.criterion
                ),
                "select StoppingCriterion::proximal_gradient_mapping",
            ));
        };
        if !absolute.is_finite() || absolute <= 0.0 {
            return Err(invalid(
                "proximal-gradient mapping tolerance must be finite and strictly positive",
            ));
        }
        if self.max_iterations == 0 {
            return Err(invalid("max_iterations must be strictly positive"));
        }
        if !self.initial_step.is_finite() || self.initial_step <= 0.0 {
            return Err(invalid("initial step must be finite and strictly positive"));
        }
        Ok(absolute)
    }
}

impl<V> Default for ProximalGradient<V> {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) struct Solution {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub termination: Termination,
    pub iterations: usize,
    pub objective: f64,
    pub stopping_criterion: StoppingCriterion,
    pub stopping_value: f64,
    pub stopping_threshold: f64,
    pub step: f64,
    pub certificate: Option<crate::DualCertificate>,
    pub screening: crate::ScreeningDiagnostics,
}

impl<M, D, P, V> Solver<Problem<&M, D, P>> for ProximalGradient<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    D: SmoothDatafit,
    P: ProximalPenalty,
    V: ProximalVector,
{
    type Fit = ProximalFit;
    type Error = FitError<M::Error>;

    fn solve(&self, problem: &Problem<&M, D, P>) -> Result<Self::Fit, Self::Error> {
        let threshold = self.validate()?;
        if problem.datafit().nobs() != problem.design().nrows() {
            return Err(invalid(
                "datafit observation count must match training rows",
            ));
        }
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        let solution = iterate(
            &mut design,
            problem.datafit(),
            problem.penalty(),
            self,
            threshold,
        )?;
        ProximalFit::from_solution(solution, design.into_preprocessing())
    }
}

impl<M, D, P, V> ObservedSolver<Problem<&M, D, P>> for ProximalGradient<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    D: SmoothDatafit,
    P: ProximalPenalty,
    V: ProximalVector,
{
    fn solve_from_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &Problem<&M, D, P>,
        initial: Option<crate::InitialPoint<'_>>,
        observer: &mut O,
    ) -> Result<Self::Fit, Self::Error> {
        let threshold = self.validate()?;
        if problem.datafit().nobs() != problem.design().nrows() {
            return Err(invalid(
                "datafit observation count must match training rows",
            ));
        }
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        let mut work = Workspace::new(design.nrows(), design.ncols());
        if let Some(initial) = initial {
            initial.apply(
                &mut work.coefficients,
                &mut work.intercept,
                problem.datafit().fits_intercept(),
            )?;
        }
        let solution = iterate_with_workspace_observed(
            &mut design,
            problem.datafit(),
            problem.penalty(),
            self,
            threshold,
            &mut work,
            observer,
        )?;
        ProximalFit::from_solution(solution, design.into_preprocessing())
    }
}

// Slice buffers keep scalar loops concrete even when components are erased.
// Native product buffers belong to the design adapter.
pub(crate) struct Workspace {
    pub(crate) coefficients: Vec<f64>,
    pub(crate) intercept: f64,
    pub(crate) eta: Vec<f64>,
    predictor_gradient: Vec<f64>,
    gradient: Vec<f64>,
    input: Vec<f64>,
    candidate: Vec<f64>,
    full_candidate: Vec<f64>,
    screening: crate::screening::ScreeningWorkspace,
}

impl Workspace {
    pub(crate) fn new(n: usize, p: usize) -> Self {
        Self {
            coefficients: vec![0.0; p],
            intercept: 0.0,
            eta: vec![0.0; n],
            predictor_gradient: vec![0.0; n],
            gradient: vec![0.0; p],
            input: vec![0.0; p],
            candidate: vec![0.0; p],
            full_candidate: Vec::new(),
            screening: crate::screening::ScreeningWorkspace::new(n, p),
        }
    }

    fn predict<O: ProximalDesign + ?Sized>(
        &mut self,
        design: &mut O,
        coefficients: &[f64],
        intercept: f64,
        iteration: usize,
    ) -> Result<(), FitError<O::Error>> {
        self.screening
            .forward(design, coefficients, &mut self.eta)?;
        for value in &mut self.eta {
            *value = finite(
                *value + intercept,
                "computing the linear predictor",
                iteration,
            )?;
        }
        Ok(())
    }
}

pub(crate) fn iterate<O, D, P, V>(
    design: &mut O,
    datafit: &D,
    penalty: &P,
    options: &ProximalGradient<V>,
    threshold: f64,
) -> Result<Solution, FitError<O::Error>>
where
    O: ProximalDesign + ?Sized,
    D: SmoothDatafit + ?Sized,
    P: ProximalPenalty + ?Sized,
{
    iterate_observed(
        design,
        datafit,
        penalty,
        options,
        threshold,
        &mut NoObserver,
    )
}

pub(crate) fn iterate_observed<O, D, P, V, T>(
    design: &mut O,
    datafit: &D,
    penalty: &P,
    options: &ProximalGradient<V>,
    threshold: f64,
    observer: &mut T,
) -> Result<Solution, FitError<O::Error>>
where
    T: SolverObserver + ?Sized,
    O: ProximalDesign + ?Sized,
    D: SmoothDatafit + ?Sized,
    P: ProximalPenalty + ?Sized,
{
    let mut work = Workspace::new(design.nrows(), design.ncols());
    iterate_with_workspace_observed(
        design, datafit, penalty, options, threshold, &mut work, observer,
    )
}

pub(crate) fn iterate_path<O, D, P, V>(
    design: &mut O,
    datafit: &D,
    penalties: &[P],
    options: &ProximalGradient<V>,
    threshold: f64,
) -> Result<Vec<Solution>, FitError<O::Error>>
where
    O: ProximalDesign + ?Sized,
    D: SmoothDatafit + ?Sized,
    P: ProximalPenalty,
{
    let mut work = Workspace::new(design.nrows(), design.ncols());
    penalties
        .iter()
        .map(|penalty| {
            iterate_with_workspace(design, datafit, penalty, options, threshold, &mut work)
        })
        .collect()
}

pub(crate) fn iterate_with_workspace<O, D, P, V>(
    design: &mut O,
    datafit: &D,
    penalty: &P,
    options: &ProximalGradient<V>,
    threshold: f64,
    work: &mut Workspace,
) -> Result<Solution, FitError<O::Error>>
where
    O: ProximalDesign + ?Sized,
    D: SmoothDatafit + ?Sized,
    P: ProximalPenalty + ?Sized,
{
    iterate_with_workspace_observed(
        design,
        datafit,
        penalty,
        options,
        threshold,
        work,
        &mut NoObserver,
    )
}

pub(crate) fn iterate_with_workspace_observed<O, D, P, V, T>(
    design: &mut O,
    datafit: &D,
    penalty: &P,
    options: &ProximalGradient<V>,
    threshold: f64,
    work: &mut Workspace,
    observer: &mut T,
) -> Result<Solution, FitError<O::Error>>
where
    T: SolverObserver + ?Sized,
    O: ProximalDesign + ?Sized,
    D: SmoothDatafit + ?Sized,
    P: ProximalPenalty + ?Sized,
{
    let mut coefficients = std::mem::take(&mut work.coefficients);
    work.screening.reset();
    let mut intercept = work.intercept;
    let mut step = options.initial_step;
    let mut iteration = 0;
    notify(observer, &coefficients, intercept, 0, StepKind::Initial);
    loop {
        work.predict(design, &coefficients, intercept, iteration)?;
        let loss = datafit
            .value(&work.eta)
            .map_err(|error| FitError::from_component(error, iteration))?;
        let mut objective = finite(
            loss + penalty
                .value(&coefficients)
                .map_err(|error| FitError::from_component(error, iteration))?,
            "computing the objective",
            iteration,
        )?;
        if options.screening && iteration % 10 == 0 {
            work.screening.check(
                design,
                datafit,
                penalty,
                &coefficients,
                &work.eta,
                objective,
                iteration,
                true,
            )?;
        }
        datafit
            .gradient(&work.eta, &mut work.predictor_gradient)
            .map_err(|error| FitError::from_component(error, iteration))?;
        for &value in &work.predictor_gradient {
            finite(value, "checking the predictor gradient", iteration)?;
        }
        work.screening
            .transpose(design, &work.predictor_gradient, &mut work.gradient)?;
        let intercept_gradient = if datafit.fits_intercept() {
            finite(
                work.predictor_gradient.iter().sum(),
                "computing the intercept gradient",
                iteration,
            )?
        } else {
            0.0
        };
        let mut accepted = None;
        for backtrack in 0..=options.max_backtracks {
            for (j, (&coefficient, value)) in coefficients.iter().zip(&mut work.input).enumerate() {
                *value = finite(
                    coefficient - step * work.gradient[j],
                    "taking a gradient step",
                    iteration,
                )?;
            }
            work.screening
                .prox(penalty, &work.input, step, &mut work.candidate)
                .map_err(|error| FitError::from_component(error, iteration))?;
            let candidate_intercept = finite(
                intercept - step * intercept_gradient,
                "updating the intercept",
                iteration,
            )?;
            let intercept_delta = candidate_intercept - intercept;
            let mut linear = intercept_gradient * intercept_delta;
            let mut quadratic = intercept_delta * (intercept_delta / step);
            let mut mapping = intercept_gradient.abs();
            for (j, (&current, &candidate)) in coefficients.iter().zip(&work.candidate).enumerate()
            {
                let delta = finite(
                    candidate - current,
                    "computing a proximal update",
                    iteration,
                )?;
                let entry = finite(
                    delta / step,
                    "computing the proximal-gradient mapping",
                    iteration,
                )?;
                mapping = mapping.max(entry.abs());
                linear += work.gradient[j] * delta;
                quadratic += delta * entry;
            }
            // Temporarily take the candidate out to keep all storage reusable
            // while a forward product mutably borrows the workspace.
            let candidate = std::mem::take(&mut work.candidate);
            let predicted = work.predict(design, &candidate, candidate_intercept, iteration);
            work.candidate = candidate;
            predicted?;
            let candidate_loss = datafit
                .value(&work.eta)
                .map_err(|error| FitError::from_component(error, iteration))?;
            let bound = finite(
                loss + linear + 0.5 * quadratic,
                "checking the smooth loss bound",
                iteration,
            )?;
            // Loss evaluation and the bound can round in opposite directions
            // near a fixed point; this slack is limited to arithmetic precision.
            let slack = 32.0 * f64::EPSILON * loss.abs().max(bound.abs());
            if candidate_loss <= bound + slack {
                accepted = Some((candidate_intercept, mapping));
                break;
            }
            if backtrack < options.max_backtracks {
                step *= 0.5;
                if step == 0.0 {
                    break;
                }
            }
        }
        let Some((candidate_intercept, mut mapping)) = accepted else {
            return Err(FitError::LineSearchFailure { iteration, step });
        };
        // The mapping belongs to the current coefficients, not the trial
        // point. Check it before accepting the next update, including at budget.
        if (mapping <= threshold || iteration == options.max_iterations)
            && work.screening.diagnostics().screened_features > 0
        {
            // A restricted fixed point must also pass the original mapping.
            design.forward(&coefficients, &mut work.eta)?;
            for value in &mut work.eta {
                *value += intercept;
            }
            datafit
                .gradient(&work.eta, &mut work.predictor_gradient)
                .map_err(|e| FitError::from_component(e, iteration))?;
            design.transpose(&work.predictor_gradient, &mut work.gradient)?;
            for ((input, &coefficient), &gradient) in
                work.input.iter_mut().zip(&coefficients).zip(&work.gradient)
            {
                *input = coefficient - step * gradient;
            }
            work.full_candidate.resize(coefficients.len(), 0.0);
            penalty
                .prox(&work.input, step, &mut work.full_candidate)
                .map_err(|e| FitError::from_component(e, iteration))?;
            mapping = if datafit.fits_intercept() {
                work.predictor_gradient.iter().sum::<f64>().abs()
            } else {
                0.0
            };
            for (&current, &candidate) in coefficients.iter().zip(&work.full_candidate) {
                mapping = mapping.max(((current - candidate) / step).abs());
            }
            objective = datafit
                .value(&work.eta)
                .map_err(|e| FitError::from_component(e, iteration))?
                + penalty
                    .value(&coefficients)
                    .map_err(|e| FitError::from_component(e, iteration))?;
        }
        if mapping <= threshold || iteration == options.max_iterations {
            if options.screening {
                // The line search leaves predictor storage at its trial point.
                design.forward(&coefficients, &mut work.eta)?;
                for value in &mut work.eta {
                    *value += intercept;
                }
                work.screening.check(
                    design,
                    datafit,
                    penalty,
                    &coefficients,
                    &work.eta,
                    objective,
                    iteration,
                    false,
                )?;
            }
            notify(
                observer,
                &coefficients,
                intercept,
                iteration,
                StepKind::Final(if mapping <= threshold {
                    Termination::Converged
                } else {
                    Termination::IterationLimit
                }),
            );
            work.coefficients = coefficients;
            work.intercept = intercept;
            return Ok(Solution {
                coefficients: work.coefficients.clone(),
                intercept,
                termination: if mapping <= threshold {
                    Termination::Converged
                } else {
                    Termination::IterationLimit
                },
                iterations: iteration,
                objective,
                stopping_criterion: options.criterion,
                stopping_value: mapping,
                stopping_threshold: threshold,
                step,
                certificate: work.screening.certificate_owned(),
                screening: work.screening.diagnostics(),
            });
        }
        std::mem::swap(&mut coefficients, &mut work.candidate);
        intercept = candidate_intercept;
        iteration += 1;
        notify(
            observer,
            &coefficients,
            intercept,
            iteration,
            StepKind::ProximalGradient,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gaussian, Preprocessing, Ridge};
    use std::convert::Infallible;

    struct Identity;
    impl ProximalDesign for Identity {
        type Error = Infallible;
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn forward(
            &mut self,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), FitError<Infallible>> {
            output.copy_from_slice(input);
            Ok(())
        }
        fn transpose(
            &mut self,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), FitError<Infallible>> {
            output.copy_from_slice(input);
            Ok(())
        }
        fn into_preprocessing(self: Box<Self>) -> Preprocessing {
            unreachable!()
        }
    }

    #[test]
    fn path_points_retain_all_iteration_buffers_and_warm_start() {
        let mut design = Identity;
        let mut work = Workspace::new(2, 2);
        let fixed = (
            work.eta.as_ptr(),
            work.predictor_gradient.as_ptr(),
            work.gradient.as_ptr(),
            work.input.as_ptr(),
        );
        let mut rotating = [work.coefficients.as_ptr(), work.candidate.as_ptr()];
        rotating.sort();
        let datafit = Gaussian::new(&[1.0, 3.0]).fit_intercept(false);
        let options = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
        let first = iterate_with_workspace(
            &mut design,
            &datafit,
            &Ridge::new(0.2),
            &options,
            1e-10,
            &mut work,
        )
        .unwrap();
        assert!(first.iterations > 0);
        let second = iterate_with_workspace(
            &mut design,
            &datafit,
            &Ridge::new(0.2),
            &options,
            1e-10,
            &mut work,
        )
        .unwrap();
        assert_eq!(second.iterations, 0);
        assert_eq!(first.coefficients, second.coefficients);
        assert_eq!(
            fixed,
            (
                work.eta.as_ptr(),
                work.predictor_gradient.as_ptr(),
                work.gradient.as_ptr(),
                work.input.as_ptr()
            )
        );
        let mut after = [work.coefficients.as_ptr(), work.candidate.as_ptr()];
        after.sort();
        assert_eq!(rotating, after);
    }
}
