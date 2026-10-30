//! Owned fitted parameters and proximal-gradient diagnostics.

use lazymatrix::RawColumns;

use super::{Preprocessing, Termination, back_transform, predict};
use crate::{LassoError, StoppingCriterion};

/// Owned original-scale parameters and proximal-gradient diagnostics.
///
/// The objective and mapping use optimization-scale coefficients. Supported
/// screened fits additionally report an original-problem dual certificate.
#[derive(Clone, Debug)]
#[must_use]
pub struct ProximalFit {
    coefficients: Vec<f64>,
    intercept: f64,
    preprocessing: Preprocessing,
    termination: Termination,
    iterations: usize,
    objective: f64,
    stopping_criterion: StoppingCriterion,
    stopping_value: f64,
    stopping_threshold: f64,
    step: f64,
    certificate: Option<crate::DualCertificate>,
    screening: crate::ScreeningDiagnostics,
}

impl ProximalFit {
    /// Feasible observation-scale dual vector, when screening was supported.
    pub fn dual_certificate(&self) -> Option<&[f64]> {
        self.certificate
            .as_ref()
            .map(crate::DualCertificate::dual_vector)
    }

    /// Original-problem dual objective, when a certificate was constructed.
    pub fn dual_objective(&self) -> Option<f64> {
        self.certificate
            .as_ref()
            .map(crate::DualCertificate::dual_objective)
    }

    /// Padded original-problem duality gap, independently of stopping choice.
    pub fn duality_gap(&self) -> Option<f64> {
        self.certificate
            .as_ref()
            .map(crate::DualCertificate::duality_gap)
    }

    /// Work and dimension reductions certified by Gap Safe screening.
    pub fn screening_diagnostics(&self) -> &crate::ScreeningDiagnostics {
        &self.screening
    }

    /// Original-scale coefficients, in training-column order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    /// Original-scale intercept, including the correction for centering.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    /// Training normalization metadata, reused for predictions.
    pub fn preprocessing(&self) -> &Preprocessing {
        &self.preprocessing
    }
    /// Whether the fit converged or exhausted its iteration budget.
    pub fn termination(&self) -> Termination {
        self.termination
    }
    /// Number of accepted proximal updates.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    /// Final normalized loss plus the complete optimization-scale penalty.
    pub fn objective(&self) -> f64 {
        self.objective
    }
    /// Criterion selected for this fit.
    pub fn stopping_criterion(&self) -> StoppingCriterion {
        self.stopping_criterion
    }
    /// Infinity norm of `(beta - prox(beta - step * gradient)) / step`,
    /// maximized with the absolute unpenalized intercept derivative.
    /// Evaluated at the returned parameters, using [`Self::step_size`].
    pub fn stopping_value(&self) -> f64 {
        self.stopping_value
    }
    /// Absolute mapping threshold used to certify convergence.
    pub fn stopping_threshold(&self) -> f64 {
        self.stopping_threshold
    }
    /// Step that passed the final smooth quadratic-bound check.
    pub fn step_size(&self) -> f64 {
        self.step
    }

    /// Predict from raw columns using original-scale parameters.
    ///
    /// # Errors
    /// Reject feature-count mismatches, invalid entries, or nonfinite arithmetic.
    pub fn predict<M: RawColumns<f64> + ?Sized>(&self, x: &M) -> Result<Vec<f64>, LassoError> {
        predict(&self.coefficients, self.intercept, self.iterations, x)
    }

    pub(crate) fn from_solution<E>(
        solution: crate::solver::proximal::Solution,
        preprocessing: Preprocessing,
    ) -> Result<Self, LassoError<E>> {
        let (coefficients, intercept) = back_transform(
            solution.coefficients,
            solution.intercept,
            &preprocessing,
            solution.iterations,
        )?;
        Ok(Self {
            coefficients,
            intercept,
            preprocessing,
            termination: solution.termination,
            iterations: solution.iterations,
            objective: solution.objective,
            stopping_criterion: solution.stopping_criterion,
            stopping_value: solution.stopping_value,
            stopping_threshold: solution.stopping_threshold,
            step: solution.step,
            certificate: solution.certificate,
            screening: solution.screening,
        })
    }
}
