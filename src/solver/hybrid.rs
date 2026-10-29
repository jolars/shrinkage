//! Quadratic PG transitions interleaved with exact cluster or segment sweeps.

use super::working_set::{WorkingSetPolicy, Workspace as WorkingSet};
mod cluster;
mod irls;
mod path;
mod quadratic;
#[cfg(test)]
mod screening_tests;
mod segment;
mod slope;
#[cfg(test)]
mod tests;

use super::observer::{InnerObserver, NoObserver, notify};
use super::{ObservedSolver, SolverObserver, StepKind};
use std::marker::PhantomData;

use quadratic::{Model, Weights};

use lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, MatrixShape, RawColumns};

use crate::error::{finite, invalid, unsupported};
use crate::screening::ScreeningWorkspace;
use crate::solver::proximal::{NativeDesign, Solution as ProximalSolution};
use crate::{
    BackendError, ChainFusedLasso, ClusteredLasso, FitError, Gaussian, HybridFit, Problem,
    ProximalDesign, ProximalVector, RuntimeProblem, Slope, SmoothDatafit, Solver,
    StoppingCriterion, Termination,
};

trait HybridPenalty: crate::GapSafePenalty + Sized {
    type State: Partition<Self>;
}

trait Partition<P>: Sized {
    fn new(n: usize, p: usize) -> Self;
    fn set_weights(&mut self, weights: &[f64]);
    fn set_screened(&mut self, zeroed: &[bool]);
    fn rebuild<M: RawColumns<f64> + lazymatrix::MatrixErrorType>(
        &mut self,
        matrix: &lazymatrix::LazyMatrix<M>,
        coefficients: &[f64],
        residual: &[f64],
        penalty: &P,
        iteration: usize,
    ) -> Result<(), FitError<M::Error>>;
    #[cfg(test)]
    fn sweep<E>(
        &mut self,
        coefficients: &mut [f64],
        intercept: &mut f64,
        fit_intercept: bool,
        penalty: &P,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        self.sweep_observed(
            coefficients,
            intercept,
            fit_intercept,
            penalty,
            iteration,
            &mut NoObserver,
        )
    }
    fn sweep_observed<E, T: SolverObserver + ?Sized>(
        &mut self,
        coefficients: &mut [f64],
        intercept: &mut f64,
        fit_intercept: bool,
        penalty: &P,
        iteration: usize,
        observer: &mut T,
    ) -> Result<(), FitError<E>>;
    fn sweep_visits(&self) -> (usize, usize);
    fn labels(&self) -> &[usize];
    fn values(&self) -> Vec<f64>;
}

impl HybridPenalty for ClusteredLasso {
    type State = cluster::State;
}

impl HybridPenalty for ChainFusedLasso {
    type State = segment::State;
}

impl HybridPenalty for Slope {
    type State = slope::State;
}

/// Curvature policy for [`Hybrid`]'s proximal-gradient transitions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HybridStep {
    /// Validate each transition against the current quadratic model.
    /// Accepted curvatures can only increase within a quadratic solve.
    Backtracking {
        /// Positive curvature seed, or the largest averaged column curvature.
        /// Include the intercept curvature when enabled; use one at zero.
        initial_curvature: Option<f64>,
        /// Maximum number of curvature doublings after the first trial.
        max_backtracks: usize,
    },
    /// Use a computed Frobenius upper bound, including the intercept column.
    /// The bound is conservative and requires no spectral estimate or Gram.
    FixedBound,
}

impl Default for HybridStep {
    fn default() -> Self {
        Self::Backtracking {
            initial_curvature: None,
            max_backtracks: 100,
        }
    }
}

/// Dense Gaussian and logistic hybrid for clustered lasso, chain fused lasso,
/// and SLOPE.
///
/// Begin with a PG transition on the working set, then alternate five cluster
/// sweeps with one PG transition. PG can split zero and nonzero blocks; CD
/// moves entire exact-equality blocks and merges exact ties. Only a fresh
/// full-space PG mapping, including the intercept derivative, certifies
/// convergence. Diagnostic calls never advance or alter the iteration.
/// Chain blocks are contiguous segments, with zero-weight edges kept as cuts.
/// SLOPE blocks group exact magnitudes while retaining individual signs. Its
/// zero block stays fixed during sweeps and can reactivate through full PG.
/// Binomial problems use IRLS with weighted hybrid inner solves and Armijo
/// backtracking on the original objective. Only the original-loss mapping
/// certifies logistic convergence. Predictions remain linear predictors.
/// Dynamic Gap Safe zero screening is enabled by default. Restrictions retain
/// the complete original penalty, and certificates always use the original
/// loss. Working sets are independently enabled through [`Self::working_set_policy`].
/// Disable both restrictions for complete matrix sweeps.
///
/// `V` is the backend's native vector, as for [`crate::ProximalGradient`].
/// Aggregates use dense storage: `O(n * K + p + n)` for `K` clusters, retaining
/// buffers at their high-water capacities. Staging a PG rebuild retains both
/// old and candidate aggregates. No normalized design or Gram is formed.
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::{Col, Mat};
/// use shrinkage::{ClusteredLasso, Gaussian, Hybrid, Problem};
/// let x = Mat::from_fn(4, 2, |i, j| ((i + 2 * j) % 3) as f64);
/// let fit = Problem::new(&x, Gaussian::new(&[1.0, 2.0, 0.0, 3.0]),
///     ClusteredLasso::new(0.1, 0.2))
///     .fit_with(&Hybrid::<Col<f64>>::new())?;
/// let predictions = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Hybrid<V = Vec<f64>> {
    criterion: StoppingCriterion,
    max_iterations: usize,
    max_outer_iterations: usize,
    max_inner_iterations: usize,
    max_outer_backtracks: usize,
    cluster_sweeps: usize,
    check_every: usize,
    screening: bool,
    working_set_policy: WorkingSetPolicy,
    step: HybridStep,
    vector: PhantomData<V>,
}

impl<V> Hybrid<V> {
    /// Use mapping tolerance `1e-6`, 10,000 iterations, five sweeps per PG,
    /// checks every ten iterations, backtracking, and safe zero screening.
    /// Logistic fits allow 100
    /// outer steps, 1,000 inner iterations per model, and 100 step halvings.
    pub fn new() -> Self {
        Self {
            criterion: StoppingCriterion::proximal_gradient_mapping(1e-6),
            max_iterations: 10_000,
            max_outer_iterations: 100,
            max_inner_iterations: 1_000,
            max_outer_backtracks: 100,
            cluster_sweeps: 5,
            check_every: 10,
            screening: true,
            working_set_policy: WorkingSetPolicy::Auto,
            step: HybridStep::default(),
            vector: PhantomData,
        }
    }

    /// Set the absolute infinity-norm full PG mapping tolerance.
    pub fn tolerance(mut self, absolute: f64) -> Self {
        self.criterion = StoppingCriterion::proximal_gradient_mapping(absolute);
        self
    }

    /// Select a stopping rule. Only absolute PG mapping is supported.
    pub fn terminate_on(mut self, criterion: StoppingCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Limit the total number of committed PG transitions and complete sweeps.
    /// Logistic fits share this budget across all quadratic subproblems.
    pub fn max_iterations(mut self, count: usize) -> Self {
        self.max_iterations = count;
        self
    }

    /// Limit accepted logistic IRLS steps (default 100). Gaussian fits ignore it.
    pub fn max_outer_iterations(mut self, count: usize) -> Self {
        self.max_outer_iterations = count;
        self
    }

    /// Limit PG transitions plus sweeps per logistic quadratic model (default
    /// 1,000). Inexact models can still provide accepted descent directions.
    /// Gaussian fits ignore this limit. The total `max_iterations` also applies.
    pub fn max_inner_iterations(mut self, count: usize) -> Self {
        self.max_inner_iterations = count;
        self
    }

    /// Limit logistic step halvings after the full trial (default 100).
    /// This differs from curvature backtracking in [`HybridStep`].
    pub fn max_outer_backtracks(mut self, count: usize) -> Self {
        self.max_outer_backtracks = count;
        self
    }

    /// Set the positive number of complete sweeps between PG transitions.
    pub fn cluster_sweeps(mut self, count: usize) -> Self {
        self.cluster_sweeps = count;
        self
    }

    /// Check every positive number of iterations, also initially and at budget.
    pub fn check_every(mut self, count: usize) -> Self {
        self.check_every = count;
        self
    }

    /// Enable dynamic Gap Safe zero screening, enabled by default.
    /// Certificates and stopping checks use the original loss and full design.
    /// Logistic quadratic models retain restrictions certified at outer iterates.
    pub fn screening(mut self, enabled: bool) -> Self {
        self.screening = enabled;
        self
    }

    /// Select backtracking or a computed fixed curvature upper bound.
    pub fn step_policy(mut self, step: HybridStep) -> Self {
        self.step = step;
        self
    }

    /// Enable automatic working sets, or disable heuristic restrictions.
    pub fn working_sets(self, enabled: bool) -> Self {
        self.working_set_policy(WorkingSetPolicy::enabled(enabled))
    }

    /// Select the working-set policy. The last working-set setter wins.
    pub fn working_set_policy(mut self, policy: WorkingSetPolicy) -> Self {
        self.working_set_policy = policy;
        self
    }

    fn validate<E>(&self) -> Result<f64, FitError<E>> {
        self.working_set_policy.validate()?;
        let StoppingCriterion::ProximalGradientMapping { absolute } = self.criterion else {
            return Err(unsupported(
                "Hybrid",
                "only full proximal-gradient mapping stopping is implemented",
                "select StoppingCriterion::proximal_gradient_mapping or Hybrid::tolerance",
            ));
        };
        if !absolute.is_finite() || absolute <= 0.0 {
            return Err(invalid(
                "proximal-gradient mapping tolerance must be finite and strictly positive",
            ));
        }
        if self.max_iterations == 0 || self.cluster_sweeps == 0 || self.check_every == 0 {
            return Err(invalid(
                "hybrid iteration limit, cluster sweeps, and check interval must be positive",
            ));
        }
        if let HybridStep::Backtracking {
            initial_curvature: Some(value),
            ..
        } = self.step
        {
            if !value.is_finite() || value <= 0.0 || !(1.0 / value).is_finite() {
                return Err(invalid(
                    "initial curvature and its reciprocal must be finite and positive",
                ));
            }
        }
        Ok(absolute)
    }
}

impl<V> Default for Hybrid<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, V> Solver<Problem<&M, Gaussian<'_>, ClusteredLasso>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
{
    type Fit = HybridFit;
    type Error = FitError<M::Error>;

    fn solve(
        &self,
        problem: &Problem<&M, Gaussian<'_>, ClusteredLasso>,
    ) -> Result<HybridFit, Self::Error> {
        solve(problem, self)
    }
}

impl<M, V> ObservedSolver<Problem<&M, Gaussian<'_>, ClusteredLasso>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
{
    fn solve_from_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &Problem<&M, Gaussian<'_>, ClusteredLasso>,
        initial: Option<crate::InitialPoint<'_>>,
        observer: &mut O,
    ) -> Result<HybridFit, Self::Error> {
        solve_observed(problem, self, initial, observer)
    }
}

impl<M, V> Solver<Problem<&M, Gaussian<'_>, ChainFusedLasso>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
{
    type Fit = HybridFit;
    type Error = FitError<M::Error>;

    fn solve(
        &self,
        problem: &Problem<&M, Gaussian<'_>, ChainFusedLasso>,
    ) -> Result<HybridFit, Self::Error> {
        solve(problem, self)
    }
}

impl<M, V> ObservedSolver<Problem<&M, Gaussian<'_>, ChainFusedLasso>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
{
    fn solve_from_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &Problem<&M, Gaussian<'_>, ChainFusedLasso>,
        initial: Option<crate::InitialPoint<'_>>,
        observer: &mut O,
    ) -> Result<HybridFit, Self::Error> {
        solve_observed(problem, self, initial, observer)
    }
}

impl<M, V> Solver<Problem<&M, Gaussian<'_>, Slope>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
{
    type Fit = HybridFit;
    type Error = FitError<M::Error>;

    fn solve(&self, problem: &Problem<&M, Gaussian<'_>, Slope>) -> Result<HybridFit, Self::Error> {
        solve(problem, self)
    }
}

impl<M, V> ObservedSolver<Problem<&M, Gaussian<'_>, Slope>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
{
    fn solve_from_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &Problem<&M, Gaussian<'_>, Slope>,
        initial: Option<crate::InitialPoint<'_>>,
        observer: &mut O,
    ) -> Result<HybridFit, Self::Error> {
        solve_observed(problem, self, initial, observer)
    }
}

fn solve<M, V, P>(
    problem: &Problem<&M, Gaussian<'_>, P>,
    options: &Hybrid<V>,
) -> Result<HybridFit, FitError<M::Error>>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
    P: HybridPenalty,
{
    solve_observed(problem, options, None, &mut NoObserver)
}

fn solve_observed<M, V, P, T>(
    problem: &Problem<&M, Gaussian<'_>, P>,
    options: &Hybrid<V>,
    initial: Option<crate::InitialPoint<'_>>,
    observer: &mut T,
) -> Result<HybridFit, FitError<M::Error>>
where
    T: SolverObserver + ?Sized,
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    V: ProximalVector,
    P: HybridPenalty,
{
    let threshold = options.validate()?;
    problem.datafit().validate(problem.design().nrows())?;
    let mut work = Workspace::<P>::new(problem.design().nrows(), problem.design().ncols());
    problem
        .penalty()
        .validate_input(&work.coefficients)
        .map_err(|e| FitError::from_component(e, 0))?;
    let prepared = problem.prepare(problem.datafit().fit_intercept)?;
    let mut design = NativeDesign::<_, V>::new_screenable(prepared);
    if problem.datafit().fit_intercept {
        work.intercept = finite(
            problem
                .datafit()
                .response
                .iter()
                .map(|y| y / design.nrows() as f64)
                .sum(),
            "initializing the hybrid intercept",
            0,
        )?;
    }
    if let Some(initial) = initial {
        initial.apply(
            &mut work.coefficients,
            &mut work.intercept,
            problem.datafit().fit_intercept,
        )?;
    }
    let solution = iterate_observed(
        &mut design,
        *problem.datafit(),
        problem.penalty(),
        options,
        threshold,
        &mut work,
        observer,
    )?;
    HybridFit::from_solution(solution, design.into_preprocessing())
}

impl<V> Solver<RuntimeProblem<'_>> for Hybrid<V> {
    type Fit = HybridFit;
    type Error = FitError<BackendError>;

    fn solve(&self, _: &RuntimeProblem<'_>) -> Result<HybridFit, Self::Error> {
        Err(unsupported(
            "Hybrid",
            "erased proximal oracles do not expose cluster or segment columns",
            "use a typed Gaussian or Binomial problem with ClusteredLasso, ChainFusedLasso, or Slope, or runtime ProximalGradient",
        ))
    }
}

pub(crate) struct Solution {
    pub proximal: ProximalSolution,
    pub labels: Vec<usize>,
    pub values: Vec<f64>,
    pub pg_steps: usize,
    pub coordinate_sweeps: usize,
    pub diagnostic_checks: usize,
    pub backtracks: usize,
    pub step_policy: HybridStep,
    pub outer_iterations: usize,
    pub outer_backtracks: usize,
    pub working_set: crate::WorkingSetDiagnostics,
}

#[derive(Clone, Copy)]
enum Selection {
    Full,
    Screened,
    Working,
}

struct Workspace<P: HybridPenalty = ClusteredLasso> {
    coefficients: Vec<f64>,
    intercept: f64,
    residual: Vec<f64>,
    derivative: Vec<f64>,
    gradient: Vec<f64>,
    input: Vec<f64>,
    candidate: Vec<f64>,
    direction: Vec<f64>,
    product: Vec<f64>,
    state: P::State,
    pending: P::State,
    weights: Weights,
    loss: f64,
    reference_curvature: Option<f64>,
    screening: ScreeningWorkspace,
    working: WorkingSet,
    path: bool,
}

impl<P: HybridPenalty> Workspace<P> {
    fn new(n: usize, p: usize) -> Self {
        Self {
            coefficients: vec![0.0; p],
            intercept: 0.0,
            residual: vec![0.0; n],
            derivative: vec![0.0; n],
            gradient: vec![0.0; p],
            input: vec![0.0; p],
            candidate: vec![0.0; p],
            direction: vec![0.0; p],
            product: vec![0.0; n],
            state: P::State::new(n, p),
            pending: P::State::new(n, p),
            weights: Weights::default(),
            loss: 0.0,
            reference_curvature: None,
            screening: ScreeningWorkspace::new(n, p),
            working: WorkingSet::new(p),
            path: false,
        }
    }

    fn prepare<'a, O: ProximalDesign>(
        &mut self,
        design: &mut O,
        datafit: impl Into<Model<'a>>,
        iteration: usize,
    ) -> Result<f64, FitError<O::Error>> {
        self.prepare_selected(design, datafit, iteration, Selection::Working)
    }

    #[cfg(test)]
    fn prepare_with_selection<'a, O: ProximalDesign>(
        &mut self,
        design: &mut O,
        datafit: impl Into<Model<'a>>,
        iteration: usize,
        full: bool,
    ) -> Result<f64, FitError<O::Error>> {
        self.prepare_selected(
            design,
            datafit,
            iteration,
            if full {
                Selection::Full
            } else {
                Selection::Screened
            },
        )
    }

    fn prepare_selected<'a, O: ProximalDesign>(
        &mut self,
        design: &mut O,
        datafit: impl Into<Model<'a>>,
        iteration: usize,
        selection: Selection,
    ) -> Result<f64, FitError<O::Error>> {
        let columns = if matches!(selection, Selection::Working) && self.working.initialized {
            design.screening_design()
        } else {
            None
        };
        if matches!(selection, Selection::Full) {
            design.forward(&self.coefficients, &mut self.residual)?;
        } else if let Some(columns) = columns {
            columns.forward_active(&self.coefficients, &self.working.active, &mut self.residual)?;
        } else {
            self.screening
                .forward(design, &self.coefficients, &mut self.residual)?;
        }
        let datafit = datafit.into();
        if matches!(selection, Selection::Full) && matches!(datafit, Model::Gaussian(_)) {
            for (eta, &prediction) in self.product.iter_mut().zip(&self.residual) {
                *eta = finite(
                    prediction + self.intercept,
                    "refreshing screening predictors",
                    iteration,
                )?;
            }
        }
        self.loss = datafit.refresh(
            &mut self.residual,
            self.intercept,
            &mut self.derivative,
            iteration,
        )?;
        let columns = if matches!(selection, Selection::Working) && self.working.initialized {
            design.screening_design()
        } else {
            None
        };
        if matches!(selection, Selection::Full) {
            design.transpose(&self.derivative, &mut self.gradient)?;
        } else if let Some(columns) = columns {
            columns.transpose_active(&self.derivative, &self.working.active, &mut self.gradient)?;
        } else {
            self.screening
                .transpose(design, &self.derivative, &mut self.gradient)?;
        }
        for &g in &self.gradient {
            finite(g, "computing the full hybrid gradient", iteration)?;
        }
        if datafit.fits_intercept() {
            Ok(finite(
                self.derivative.iter().sum(),
                "computing the hybrid intercept gradient",
                iteration,
            )?)
        } else {
            Ok(0.0)
        }
    }

    fn trial<E>(
        &mut self,
        penalty: &P,
        curvature: f64,
        intercept_gradient: f64,
        iteration: usize,
    ) -> Result<f64, FitError<E>> {
        self.trial_with_restriction(penalty, curvature, intercept_gradient, iteration, false)
    }

    fn trial_with_restriction<E>(
        &mut self,
        penalty: &P,
        curvature: f64,
        intercept_gradient: f64,
        iteration: usize,
        restricted: bool,
    ) -> Result<f64, FitError<E>> {
        self.trial_selected(
            penalty,
            curvature,
            intercept_gradient,
            iteration,
            if restricted {
                Selection::Working
            } else {
                Selection::Full
            },
        )
    }

    fn trial_selected<E>(
        &mut self,
        penalty: &P,
        curvature: f64,
        intercept_gradient: f64,
        iteration: usize,
        selection: Selection,
    ) -> Result<f64, FitError<E>> {
        let step = finite(
            1.0 / curvature,
            "computing a hybrid PG step size",
            iteration,
        )?;
        for ((input, &coefficient), &g) in self
            .input
            .iter_mut()
            .zip(&self.coefficients)
            .zip(&self.gradient)
        {
            *input = finite(
                coefficient - step * g,
                "taking a full hybrid gradient step",
                iteration,
            )?;
        }
        if matches!(selection, Selection::Working) && self.working.initialized {
            penalty
                .prox_zeroed(&self.input, &self.working.zeroed, step, &mut self.candidate)
                .map_err(|e| FitError::from_component(e, iteration))?;
        } else if matches!(selection, Selection::Screened | Selection::Working) {
            self.screening
                .prox(penalty, &self.input, step, &mut self.candidate)
                .map_err(|e| FitError::from_component(e, iteration))?;
        } else {
            penalty
                .prox(&self.input, step, &mut self.candidate)
                .map_err(|e| FitError::from_component(e, iteration))?;
        }
        Ok(finite(
            self.intercept - step * intercept_gradient,
            "taking a hybrid intercept step",
            iteration,
        )?)
    }

    fn accepts<O: ProximalDesign>(
        &mut self,
        design: &mut O,
        curvature: f64,
        candidate_intercept: f64,
        iteration: usize,
    ) -> Result<bool, FitError<O::Error>> {
        let intercept_delta = finite(
            candidate_intercept - self.intercept,
            "forming a hybrid displacement",
            iteration,
        )?;
        let mut scale = intercept_delta.abs();
        for ((d, &next), &current) in self
            .direction
            .iter_mut()
            .zip(&self.candidate)
            .zip(&self.coefficients)
        {
            *d = finite(next - current, "forming a hybrid displacement", iteration)?;
            scale = scale.max(d.abs());
        }
        if scale == 0.0 {
            return Ok(true);
        }
        let intercept = intercept_delta / scale;
        let mut norm = intercept.abs();
        for d in &mut self.direction {
            *d /= scale;
            norm = norm.hypot(*d);
        }
        let columns = if self.working.initialized {
            design.screening_design()
        } else {
            None
        };
        if let Some(columns) = columns {
            columns.forward_active(&self.direction, &self.working.active, &mut self.product)?;
        } else {
            self.screening
                .forward(design, &self.direction, &mut self.product)?;
        }
        let image_norm = self
            .weights
            .image_norm(&self.product, intercept, iteration)?;
        Ok(image_norm <= curvature.sqrt() * norm * (1.0 + 64.0 * f64::EPSILON))
    }

    fn diagnostics<'a, O: ProximalDesign>(
        &mut self,
        design: &mut O,
        datafit: impl Into<Model<'a>>,
        penalty: &P,
        curvature: f64,
        iteration: usize,
    ) -> Result<(f64, f64), FitError<O::Error>> {
        self.diagnostics_selected(design, datafit, penalty, curvature, iteration, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn diagnostics_selected<'a, O: ProximalDesign>(
        &mut self,
        design: &mut O,
        datafit: impl Into<Model<'a>>,
        penalty: &P,
        curvature: f64,
        iteration: usize,
        complete: bool,
    ) -> Result<(f64, f64), FitError<O::Error>> {
        let datafit = datafit.into();
        let selection = if !complete {
            Selection::Working
        } else if matches!(datafit, Model::Gaussian(_)) {
            Selection::Full
        } else {
            Selection::Screened
        };
        let curvature = self.reference_curvature.unwrap_or(curvature);
        let intercept_gradient = self.prepare_selected(design, datafit, iteration, selection)?;
        self.trial_selected::<O::Error>(
            penalty,
            curvature,
            intercept_gradient,
            iteration,
            selection,
        )?;
        let mut mapping = intercept_gradient.abs();
        for (j, (&current, &candidate)) in self.coefficients.iter().zip(&self.candidate).enumerate()
        {
            let score = finite(
                (current - candidate) * curvature,
                "computing the hybrid score",
                iteration,
            )?
            .abs();
            if complete {
                self.working.scores[j] = score;
            }
            mapping = mapping.max(
                finite(
                    (current - candidate) * curvature,
                    "computing the full hybrid mapping",
                    iteration,
                )?
                .abs(),
            );
        }
        let objective = finite(
            self.loss
                + penalty
                    .value(&self.coefficients)
                    .map_err(|e| FitError::from_component(e, iteration))?,
            "computing the hybrid objective",
            iteration,
        )?;
        if complete {
            self.working.diagnostics.full_checks += 1;
        } else {
            self.working.diagnostics.restricted_checks += 1;
        }
        Ok((objective, mapping))
    }
}

fn initial_curvature<M: RawColumns<f64> + lazymatrix::MatrixErrorType, V: ProximalVector, W>(
    design: &NativeDesign<M, V>,
    fit_intercept: bool,
    weights: &Weights,
    options: &Hybrid<W>,
) -> Result<f64, FitError<M::Error>> {
    let mut maximum = if fit_intercept {
        weights.intercept_curvature(design.matrix().nrows())
    } else {
        0.0
    };
    let mut trace = maximum;
    for j in 0..design.matrix().ncols() {
        let q = finite(
            weights.column_curvature(design.matrix().column(j)),
            "computing hybrid column curvatures",
            0,
        )?;
        maximum = maximum.max(q);
        trace = finite(trace + q, "computing a hybrid Frobenius bound", 0)?;
    }
    let value = match options.step {
        HybridStep::FixedBound => {
            // The trace bounds the largest eigenvalue of the augmented
            // averaged Hessian; a rounding guard keeps the computed bound conservative.
            let guard = 1.0
                + 16.0
                    * f64::EPSILON
                    * (design.matrix().nrows() as f64 + design.matrix().ncols() as f64 + 1.0);
            finite(trace * guard, "rounding the hybrid curvature bound", 0)?
        }
        HybridStep::Backtracking {
            initial_curvature, ..
        } => initial_curvature.unwrap_or(maximum),
    };
    Ok(if value == 0.0 { 1.0 } else { value })
}

#[cfg(test)]
fn iterate_quadratic<'a, M, V, W, P>(
    design: &mut NativeDesign<M, V>,
    datafit: Model<'a>,
    penalty: &P,
    options: &Hybrid<W>,
    threshold: f64,
    work: &mut Workspace<P>,
) -> Result<Progress, FitError<M::Error>>
where
    M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
    P: HybridPenalty,
{
    iterate_quadratic_observed(
        design,
        datafit,
        penalty,
        options,
        threshold,
        work,
        &mut NoObserver,
    )
}

fn iterate_quadratic_observed<'a, M, V, W, P, T>(
    design: &mut NativeDesign<M, V>,
    datafit: Model<'a>,
    penalty: &P,
    options: &Hybrid<W>,
    threshold: f64,
    work: &mut Workspace<P>,
    observer: &mut T,
) -> Result<Progress, FitError<M::Error>>
where
    T: SolverObserver + ?Sized,
    M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
    P: HybridPenalty,
{
    datafit
        .validate(design.nrows())
        .map_err(|e| FitError::from_component(e, 0))?;
    if work.working.diagnostics.policy == WorkingSetPolicy::Auto {
        work.working
            .begin(options.working_set_policy, false, &work.coefficients);
    } else {
        work.working.begin_model(&work.coefficients);
    }
    work.weights.set(datafit.weights());
    work.state.set_weights(datafit.weights());
    work.pending.set_weights(datafit.weights());
    let mut curvature =
        initial_curvature(design, datafit.fits_intercept(), &work.weights, options)?;
    let (mut pg_steps, mut coordinate_sweeps, mut diagnostic_checks, mut backtracks) = (0, 0, 0, 0);
    let mut since_pg = options.cluster_sweeps;
    let mut request_check = true;
    let mut restricted_done = false;
    let mut moved = false;
    let mut target = threshold;
    notify(
        observer,
        &work.coefficients,
        work.intercept,
        0,
        StepKind::Initial,
    );
    for iteration in 0..=options.max_iterations {
        if request_check
            || iteration % options.check_every == 0
            || iteration == options.max_iterations
        {
            let (objective, mapping) =
                work.diagnostics(design, datafit, penalty, curvature, iteration)?;
            diagnostic_checks += 1;
            if let Model::Gaussian(original) = datafit {
                if options.screening
                    && work.screening.check(
                        design,
                        &original,
                        penalty,
                        &work.coefficients,
                        &work.product,
                        objective,
                        iteration,
                        mapping > threshold && iteration < options.max_iterations,
                    )?
                {
                    work.state.set_screened(&work.screening.zeroed);
                    work.pending.set_screened(&work.screening.zeroed);
                    since_pg = options.cluster_sweeps;
                }
            }
            if mapping <= threshold || iteration == options.max_iterations {
                work.working.select(
                    &work.coefficients,
                    &work.screening.zeroed,
                    false,
                    false,
                    threshold,
                );
                work.pending.set_screened(&work.working.zeroed);
                // A final canonical scan also covers an initially optimal
                // start and records signed values before back-transformation.
                work.pending.rebuild(
                    design.matrix(),
                    &work.coefficients,
                    &work.residual,
                    penalty,
                    iteration,
                )?;
                notify(
                    observer,
                    &work.coefficients,
                    work.intercept,
                    iteration,
                    StepKind::Final(if mapping <= threshold {
                        Termination::Converged
                    } else {
                        Termination::IterationLimit
                    }),
                );
                return Ok(Progress {
                    iterations: iteration,
                    objective,
                    mapping,
                    curvature,
                    termination: if mapping <= threshold {
                        Termination::Converged
                    } else {
                        Termination::IterationLimit
                    },
                    pg_steps,
                    coordinate_sweeps,
                    diagnostic_checks,
                    backtracks,
                });
            }
            let changed = work.working.select(
                &work.coefficients,
                &work.screening.zeroed,
                restricted_done || (iteration > 0 && !moved),
                iteration > 0 && !moved,
                threshold,
            );
            work.state.set_screened(&work.working.zeroed);
            work.pending.set_screened(&work.working.zeroed);
            if changed {
                since_pg = options.cluster_sweeps;
            }
            target = threshold.max(0.1 * mapping);
            moved = false;
        }
        let previous_intercept = work.intercept;
        work.direction.copy_from_slice(&work.coefficients);
        let moved_step;
        if since_pg == options.cluster_sweeps {
            let intercept_gradient = work.prepare(design, datafit, iteration)?;
            let max_backtracks = match options.step {
                HybridStep::Backtracking { max_backtracks, .. } => max_backtracks,
                HybridStep::FixedBound => 0,
            };
            let mut trial_curvature = curvature;
            let mut accepted = None;
            for attempt in 0..=max_backtracks {
                let trial = work.trial_with_restriction::<M::Error>(
                    penalty,
                    trial_curvature,
                    intercept_gradient,
                    iteration,
                    true,
                );
                match trial {
                    Ok(candidate_intercept) => {
                        if matches!(options.step, HybridStep::FixedBound)
                            || work.accepts(
                                design,
                                trial_curvature,
                                candidate_intercept,
                                iteration,
                            )?
                        {
                            accepted = Some(candidate_intercept);
                            break;
                        }
                    }
                    Err(error) => {
                        if !matches!(error, FitError::NumericalFailure { .. })
                            || attempt == max_backtracks
                        {
                            return Err(error);
                        }
                    }
                }
                if attempt < max_backtracks {
                    trial_curvature = finite(
                        trial_curvature * 2.0,
                        "increasing hybrid PG curvature",
                        iteration,
                    )?;
                    backtracks += 1;
                }
            }
            let Some(candidate_intercept) = accepted else {
                return Err(FitError::LineSearchFailure {
                    iteration,
                    step: 1.0 / trial_curvature,
                });
            };
            let columns = if work.working.initialized {
                design.screening_design()
            } else {
                None
            };
            if let Some(columns) = columns {
                columns.forward_active(&work.candidate, &work.working.active, &mut work.product)?;
            } else {
                work.screening
                    .forward(design, &work.candidate, &mut work.product)?;
            }
            moved_step =
                candidate_intercept != work.intercept || work.candidate != work.coefficients;
            work.working.diagnostics.feature_visits += work.working.active.len();
            datafit.refresh(
                &mut work.product,
                candidate_intercept,
                &mut work.derivative,
                iteration,
            )?;
            work.pending.rebuild(
                design.matrix(),
                &work.candidate,
                &work.product,
                penalty,
                iteration,
            )?;
            std::mem::swap(&mut work.coefficients, &mut work.candidate);
            std::mem::swap(&mut work.state, &mut work.pending);
            work.intercept = candidate_intercept;
            curvature = trial_curvature;
            pg_steps += 1;
            since_pg = 0;
            notify(
                observer,
                &work.coefficients,
                work.intercept,
                iteration + 1,
                StepKind::ProximalGradient,
            );
        } else {
            work.state.sweep_observed(
                &mut work.coefficients,
                &mut work.intercept,
                datafit.fits_intercept(),
                penalty,
                iteration,
                observer,
            )?;
            moved_step =
                previous_intercept != work.intercept || work.direction != work.coefficients;
            let (features, blocks) = work.state.sweep_visits();
            work.working.diagnostics.feature_visits += features;
            work.working.diagnostics.block_visits += blocks;
            coordinate_sweeps += 1;
            since_pg += 1;
            notify(
                observer,
                &work.coefficients,
                work.intercept,
                iteration + 1,
                StepKind::CoordinateSweep,
            );
        }
        moved |= moved_step;
        work.working.record(&work.coefficients);
        if work.working.active.len() < work.screening.active.len() {
            let (_, local) = work.diagnostics_selected(
                design,
                datafit,
                penalty,
                curvature,
                iteration + 1,
                false,
            )?;
            restricted_done = local <= target;
            request_check = restricted_done || !moved;
        } else {
            restricted_done = false;
            request_check = false;
        }
        request_check |=
            iteration == 0 && work.working.diagnostics.policy == WorkingSetPolicy::EverActive;
    }
    unreachable!("the iteration budget always triggers a final full check")
}

struct Progress {
    iterations: usize,
    objective: f64,
    mapping: f64,
    curvature: f64,
    termination: Termination,
    pg_steps: usize,
    coordinate_sweeps: usize,
    diagnostic_checks: usize,
    backtracks: usize,
}

impl Progress {
    fn finish<P: HybridPenalty, V>(
        self,
        work: &Workspace<P>,
        options: &Hybrid<V>,
        threshold: f64,
    ) -> Solution {
        Solution {
            proximal: ProximalSolution {
                coefficients: work.coefficients.clone(),
                intercept: work.intercept,
                termination: self.termination,
                iterations: self.iterations,
                objective: self.objective,
                stopping_criterion: options.criterion,
                stopping_value: self.mapping,
                stopping_threshold: threshold,
                step: 1.0 / self.curvature,
                certificate: work.screening.certificate_owned(),
                screening: work.screening.diagnostics(),
            },
            labels: work.pending.labels().to_vec(),
            values: work.pending.values(),
            pg_steps: self.pg_steps,
            coordinate_sweeps: self.coordinate_sweeps,
            diagnostic_checks: self.diagnostic_checks,
            backtracks: self.backtracks,
            step_policy: options.step,
            outer_iterations: 0,
            outer_backtracks: 0,
            working_set: work.working.diagnostics.clone(),
        }
    }
}

#[cfg(test)]
fn iterate<M, V, W, P>(
    design: &mut NativeDesign<M, V>,
    datafit: Gaussian<'_>,
    penalty: &P,
    options: &Hybrid<W>,
    threshold: f64,
    work: &mut Workspace<P>,
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
        work,
        &mut NoObserver,
    )
}

fn iterate_observed<M, V, W, P, T>(
    design: &mut NativeDesign<M, V>,
    datafit: Gaussian<'_>,
    penalty: &P,
    options: &Hybrid<W>,
    threshold: f64,
    work: &mut Workspace<P>,
    observer: &mut T,
) -> Result<Solution, FitError<M::Error>>
where
    T: SolverObserver + ?Sized,
    M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
    P: HybridPenalty,
{
    work.working
        .begin(options.working_set_policy, work.path, &work.coefficients);
    work.screening.reset();
    work.state.set_screened(&work.screening.zeroed);
    work.pending.set_screened(&work.screening.zeroed);
    Ok(iterate_quadratic_observed(
        design,
        datafit.into(),
        penalty,
        options,
        threshold,
        work,
        observer,
    )?
    .finish(work, options, threshold))
}
