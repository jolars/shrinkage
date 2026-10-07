//! Backtracking proximal gradient over concrete or object-safe design oracles.

mod design;
pub(crate) use design::NativeDesign;
pub use design::ProximalDesign;

use std::marker::PhantomData;

use lazymatrix::{
    ColumnStats, DotSlice, ElemDivAssign, MatTransposeVecInto, MatVecInto, RawColumns,
    ScaledSubSlice, SubScalarAssign, SumEntries,
};

use crate::error::{finite, invalid};
use crate::{
    FitError, Problem, ProximalFit, ProximalPenalty, SmoothDatafit, Solver, StoppingCriterion,
    Termination,
};

/// Owned CPU storage for a backend's product vectors.
///
/// Matrix products and normalization remain in LazyMatrix. This capability
/// supplies reusable buffers and bridges native vectors to the slice-based
/// predictor and proximal capabilities. It does not represent device storage.
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

impl ProximalVector for Vec<f64> {
    fn zeros(length: usize) -> Self {
        vec![0.0; length]
    }
    fn get(&self, index: usize) -> f64 {
        self[index]
    }
    fn set(&mut self, index: usize, value: f64) {
        self[index] = value;
    }
}

#[cfg(feature = "faer_v0_24")]
impl ProximalVector for faer::Col<f64> {
    fn zeros(length: usize) -> Self {
        Self::zeros(length)
    }
    fn get(&self, index: usize) -> f64 {
        self[index]
    }
    fn set(&mut self, index: usize, value: f64) {
        self[index] = value;
    }
}

#[cfg(feature = "nalgebra_v0_34")]
impl ProximalVector for nalgebra::DVector<f64> {
    fn zeros(length: usize) -> Self {
        Self::zeros(length)
    }
    fn get(&self, index: usize) -> f64 {
        self[index]
    }
    fn set(&mut self, index: usize, value: f64) {
        self[index] = value;
    }
}

#[cfg(feature = "ndarray_v0_17")]
impl ProximalVector for ndarray::Array1<f64> {
    fn zeros(length: usize) -> Self {
        Self::zeros(length)
    }
    fn get(&self, index: usize) -> f64 {
        self[index]
    }
    fn set(&mut self, index: usize, value: f64) {
        self[index] = value;
    }
}

/// Proximal gradient with backtracking for convex smooth predictor losses and
/// complete convex proximal penalties.
///
/// `V` is the design backend's owned vector: `faer::Col<f64>`,
/// `nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, or `Vec<f64>` for sprs.
/// No normalized design or Gram matrix is formed. LazyMatrix 0.5.0 still
/// clones the input to scaled forward products.
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

    pub(crate) fn validate<E>(&self) -> Result<f64, FitError<E>> {
        let StoppingCriterion::ProximalGradientMapping { absolute } = self.criterion else {
            return Err(invalid(
                "proximal gradient supports proximal-gradient mapping only; select StoppingCriterion::proximal_gradient_mapping",
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
        let mut design = NativeDesign::<_, V>::new(prepared);
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

// Slice buffers keep scalar loops concrete even when components are erased.
// Native product buffers belong to the design adapter.
struct Workspace {
    eta: Vec<f64>,
    predictor_gradient: Vec<f64>,
    gradient: Vec<f64>,
    input: Vec<f64>,
    candidate: Vec<f64>,
}

impl Workspace {
    fn new(n: usize, p: usize) -> Self {
        Self {
            eta: vec![0.0; n],
            predictor_gradient: vec![0.0; n],
            gradient: vec![0.0; p],
            input: vec![0.0; p],
            candidate: vec![0.0; p],
        }
    }

    fn predict<O: ProximalDesign + ?Sized>(
        &mut self,
        design: &mut O,
        coefficients: &[f64],
        intercept: f64,
        iteration: usize,
    ) -> Result<(), FitError<O::Error>> {
        design.forward(coefficients, &mut self.eta)?;
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
    let mut work = Workspace::new(design.nrows(), design.ncols());
    let mut coefficients = vec![0.0; design.ncols()];
    let mut intercept = 0.0;
    let mut step = options.initial_step;
    let mut iteration = 0;
    loop {
        work.predict(design, &coefficients, intercept, iteration)?;
        let loss = datafit
            .value(&work.eta)
            .map_err(|error| FitError::from_component(error, iteration))?;
        let objective = finite(
            loss + penalty
                .value(&coefficients)
                .map_err(|error| FitError::from_component(error, iteration))?,
            "computing the objective",
            iteration,
        )?;
        datafit
            .gradient(&work.eta, &mut work.predictor_gradient)
            .map_err(|error| FitError::from_component(error, iteration))?;
        for &value in &work.predictor_gradient {
            finite(value, "checking the predictor gradient", iteration)?;
        }
        design.transpose(&work.predictor_gradient, &mut work.gradient)?;
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
            penalty
                .prox(&work.input, step, &mut work.candidate)
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
        let Some((candidate_intercept, mapping)) = accepted else {
            return Err(FitError::LineSearchFailure { iteration, step });
        };
        // The mapping belongs to the current coefficients, not the trial
        // point. Check it before accepting the next update, including at budget.
        if mapping <= threshold || iteration == options.max_iterations {
            return Ok(Solution {
                coefficients,
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
            });
        }
        std::mem::swap(&mut coefficients, &mut work.candidate);
        intercept = candidate_intercept;
        iteration += 1;
    }
}
