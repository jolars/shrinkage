//! Borrowed observations of optimization states.

use super::Solver;
use crate::error::invalid;
use crate::{FitError, Termination};

/// The operation that produced an observed optimization state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepKind {
    /// Validated solver initialization, before the first update.
    Initial,
    /// One coordinate update, with a zero-based feature index.
    Coordinate {
        /// Coordinate updated by this step.
        index: usize,
    },
    /// One cluster or segment update within a hybrid sweep.
    Block {
        /// Zero-based visit within the sweep, rather than a feature index.
        index: usize,
    },
    /// An unpenalized intercept update.
    Intercept,
    /// An accepted proximal-gradient update on the current eligible features.
    ProximalGradient,
    /// One complete cluster or segment sweep in a hybrid solve.
    CoordinateSweep,
    /// An accepted outer IRLS update on the original objective.
    Irls,
    /// Final state and termination reason, including initially optimal fits.
    Final(Termination),
}

/// A borrowed optimization state, valid only during an observer callback.
///
/// Coefficients and the intercept use the solver's optimization scale. Returned
/// fits back-transform these parameters using their preprocessing metadata.
/// Iteration counts use the selected solver's documented units. Inner callbacks
/// count IRLS sweeps, Gaussian hybrid block visits, or binomial hybrid quadratic
/// iterations. Rejected line-search trials are never emitted.
#[derive(Clone, Copy, Debug)]
pub struct SolverStep<'a> {
    /// Optimization-scale coefficients, without an allocation or copy.
    pub coefficients: &'a [f64],
    /// Optimization-scale unpenalized intercept.
    pub intercept: f64,
    /// Solver iteration count at this state.
    pub iteration: usize,
    /// Operation that produced this state.
    pub kind: StepKind,
}

/// A read-only consumer of solver states.
///
/// Closures accepting [`SolverStep`] implement this trait. Copy any parameters
/// that must outlive a callback. Observers cannot alter stopping or acceptance.
pub trait SolverObserver {
    /// Observe initialization, an accepted update, or the final state.
    fn observe(&mut self, step: SolverStep<'_>);

    /// Observe an update inside an outer iteration.
    ///
    /// These states solve a surrogate or a block restriction. They need not
    /// decrease the original objective or become an accepted outer state.
    /// The default ignores them, including for closure observers.
    /// `outer_iteration` identifies the parent IRLS update or hybrid sweep.
    fn observe_inner(&mut self, _step: SolverStep<'_>, _outer_iteration: usize) {}
}

impl<F: for<'a> FnMut(SolverStep<'a>)> SolverObserver for F {
    fn observe(&mut self, step: SolverStep<'_>) {
        self(step);
    }
}

/// An explicit starting point on the solver's optimization scale.
///
/// Supply coefficients after the problem's normalization, and an intercept on
/// that same scale. A problem without an intercept requires an intercept of zero.
#[derive(Clone, Copy, Debug)]
pub struct InitialPoint<'a> {
    /// One finite coefficient per design column.
    pub coefficients: &'a [f64],
    /// Finite unpenalized intercept.
    pub intercept: f64,
}

impl InitialPoint<'_> {
    pub(crate) fn apply<E>(
        self,
        coefficients: &mut [f64],
        intercept: &mut f64,
        fits_intercept: bool,
    ) -> Result<(), FitError<E>> {
        self.validate(coefficients.len(), fits_intercept)?;
        coefficients.copy_from_slice(self.coefficients);
        *intercept = self.intercept;
        Ok(())
    }

    pub(crate) fn validate<E>(self, ncols: usize, fits_intercept: bool) -> Result<(), FitError<E>> {
        if self.coefficients.len() != ncols
            || self.coefficients.iter().any(|b| !b.is_finite())
            || !self.intercept.is_finite()
            || (!fits_intercept && self.intercept != 0.0)
        {
            return Err(invalid(
                "initial point must have one finite coefficient per column and a finite intercept, which must be zero when disabled",
            ));
        }
        Ok(())
    }
}

/// Optional observation capability for single typed fits.
pub trait ObservedSolver<P>: Solver<P> {
    /// Fit with read-only observations of accepted optimization states.
    ///
    /// # Errors
    /// Returns the same validation, numerical, and backend errors as ordinary
    /// fitting. Failed fits need not emit a final state.
    fn solve_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &P,
        observer: &mut O,
    ) -> Result<Self::Fit, Self::Error> {
        self.solve_from_with_observer(problem, None, observer)
    }

    /// Fit from an explicit optimization-scale point, or the solver's default.
    ///
    /// # Errors
    /// Also rejects nonfinite parameters, a coefficient count that differs from
    /// the design, or a nonzero intercept when the problem disables it.
    fn solve_from_with_observer<O: SolverObserver + ?Sized>(
        &self,
        problem: &P,
        initial: Option<InitialPoint<'_>>,
        observer: &mut O,
    ) -> Result<Self::Fit, Self::Error>;
}

pub(crate) struct NoObserver;
impl SolverObserver for NoObserver {
    #[inline]
    fn observe(&mut self, _: SolverStep<'_>) {}
}

#[inline]
pub(crate) fn notify<O: SolverObserver + ?Sized>(
    observer: &mut O,
    coefficients: &[f64],
    intercept: f64,
    iteration: usize,
    kind: StepKind,
) {
    observer.observe(SolverStep {
        coefficients,
        intercept,
        iteration,
        kind,
    });
}

#[inline]
pub(crate) fn notify_inner<O: SolverObserver + ?Sized>(
    observer: &mut O,
    coefficients: &[f64],
    intercept: f64,
    outer_iteration: usize,
    iteration: usize,
    kind: StepKind,
) {
    observer.observe_inner(
        SolverStep {
            coefficients,
            intercept,
            iteration,
            kind,
        },
        outer_iteration,
    );
}

pub(crate) struct InnerObserver<'a, O: ?Sized> {
    pub(crate) observer: &'a mut O,
    pub(crate) outer_iteration: usize,
}

impl<O: SolverObserver + ?Sized> SolverObserver for InnerObserver<'_, O> {
    fn observe(&mut self, step: SolverStep<'_>) {
        if !matches!(step.kind, StepKind::Initial | StepKind::Final(_)) {
            self.observer.observe_inner(step, self.outer_iteration);
        }
    }

    fn observe_inner(&mut self, step: SolverStep<'_>, outer_iteration: usize) {
        self.observer.observe_inner(
            SolverStep {
                iteration: outer_iteration,
                ..step
            },
            self.outer_iteration,
        );
    }
}
