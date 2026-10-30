//! Owned Gaussian lasso parameters and coordinate-descent diagnostics.

use lazymatrix::RawColumns;

use super::{Preprocessing, Termination, back_transform, predict};
use crate::{LassoError, StoppingCriterion};

/// Owned original-scale parameters, preprocessing metadata, and diagnostics.
///
/// Check [`Self::termination`] before treating a fit as converged. Diagnostics
/// refer to the optimization-scale penalty even though coefficients use the
/// original scale.
#[derive(Clone, Debug)]
#[must_use]
pub struct LassoFit {
    working_set: crate::WorkingSetDiagnostics,

    coefficients: Vec<f64>,
    intercept: f64,
    preprocessing: Preprocessing,
    termination: Termination,
    iterations: usize,
    objective: f64,
    kkt_violation: f64,
    dual_certificate: Vec<f64>,
    dual_objective: f64,
    duality_gap: f64,
    reference_loss: f64,
    stopping_criterion: StoppingCriterion,
    stopping_value: f64,
    stopping_threshold: f64,
}

impl LassoFit {
    /// Work performed by heuristic feature selection, separate from safe screening.
    pub fn working_set_diagnostics(&self) -> &crate::WorkingSetDiagnostics {
        &self.working_set
    }

    /// Original-scale coefficients, in training-column order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Original-scale intercept, including the correction for centering.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Fitted training-column centers and scales.
    pub fn preprocessing(&self) -> &Preprocessing {
        &self.preprocessing
    }

    /// Whether the fit converged or exhausted its iteration budget.
    pub fn termination(&self) -> Termination {
        self.termination
    }

    /// Number of coordinate sweeps over the current working set performed.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Final `||residual||² / (2n) + lambda * ||beta||₁`.
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// Maximum absolute KKT violation, as defined by [`crate::Lasso::tolerance`].
    pub fn kkt_violation(&self) -> f64 {
        self.kkt_violation
    }

    /// Feasible dual vector `u`, with `||X_tilde' u||_inf <= lambda` and
    /// `sum(u) = 0` when an intercept was fitted, up to floating-point error.
    /// At zero penalty, the certificate can be zero and its gap can be loose.
    pub fn dual_certificate(&self) -> &[f64] {
        &self.dual_certificate
    }

    /// Gaussian dual objective `y' u - n ||u||² / 2`.
    pub fn dual_objective(&self) -> f64 {
        self.dual_objective
    }

    /// Nonnegative primal minus dual objective.
    pub fn duality_gap(&self) -> f64 {
        self.duality_gap
    }

    /// Averaged loss at zero coefficients and the optimal enabled intercept.
    pub fn reference_loss(&self) -> f64 {
        self.reference_loss
    }

    /// Criterion selected for this fit.
    pub fn stopping_criterion(&self) -> StoppingCriterion {
        self.stopping_criterion
    }

    /// Final value of the selected criterion.
    pub fn stopping_value(&self) -> f64 {
        self.stopping_value
    }

    /// Threshold applied to the selected criterion.
    pub fn stopping_threshold(&self) -> f64 {
        self.stopping_threshold
    }

    /// Predict from raw columns in the same order as the training design.
    ///
    /// Uses original-scale parameters, without estimating new centers or
    /// scales. Empty prediction batches return an empty vector.
    ///
    /// # Errors
    /// Returns an error for a feature-count mismatch, nonfinite input, or
    /// nonfinite prediction arithmetic.
    pub fn predict<M: RawColumns<f64> + ?Sized>(&self, x: &M) -> Result<Vec<f64>, LassoError> {
        predict(&self.coefficients, self.intercept, self.iterations, x)
    }
}

impl LassoFit {
    pub(crate) fn from_solution<E>(
        solution: crate::solver::coordinate::Solution,
        preprocessing: Preprocessing,
    ) -> Result<Self, LassoError<E>> {
        let (coefficients, intercept) = back_transform(
            solution.coefficients,
            solution.intercept,
            &preprocessing,
            solution.iterations,
        )?;
        Ok(LassoFit {
            working_set: solution.working_set,
            coefficients,
            intercept,
            preprocessing,
            termination: solution.termination,
            iterations: solution.iterations,
            objective: solution.objective,
            kkt_violation: solution.kkt_violation,
            dual_certificate: solution.dual_certificate,
            dual_objective: solution.dual_objective,
            duality_gap: solution.duality_gap,
            reference_loss: solution.reference_loss,
            stopping_criterion: solution.stopping_criterion,
            stopping_value: solution.stopping_value,
            stopping_threshold: solution.stopping_threshold,
        })
    }
}
