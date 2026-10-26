use lazymatrix::RawColumns;

use crate::{FitError, HybridStep, Preprocessing, ProximalFit, StoppingCriterion, Termination};

/// Original-scale Gaussian parameters and full-space hybrid diagnostics.
///
/// Cluster labels and values refer to signed optimization-scale coefficients.
/// Clustered lasso orders blocks by value; chain fused lasso orders contiguous
/// segments by feature and retains zero-weight edges as segment boundaries.
/// Scaling can change equalities among the reported original-scale parameters.
/// No duality gap is reported. A sweep that leaves all blocks unchanged does
/// not certify convergence; check [`Self::termination`] and the full mapping.
#[derive(Clone, Debug)]
#[must_use]
pub struct HybridFit {
    fit: ProximalFit,
    labels: Vec<usize>,
    values: Vec<f64>,
    pg_steps: usize,
    coordinate_sweeps: usize,
    diagnostic_checks: usize,
    backtracks: usize,
    step_policy: HybridStep,
}

impl HybridFit {
    /// Original-scale coefficients in training-column order.
    pub fn coefficients(&self) -> &[f64] {
        self.fit.coefficients()
    }
    /// Original-scale intercept, including the centering correction.
    pub fn intercept(&self) -> f64 {
        self.fit.intercept()
    }
    /// Training normalization retained for interpreting optimization values.
    pub fn preprocessing(&self) -> &Preprocessing {
        self.fit.preprocessing()
    }
    /// Whether the full mapping passed or the iteration budget expired.
    pub fn termination(&self) -> Termination {
        self.fit.termination()
    }
    /// Total committed PG transitions plus complete coordinate sweeps.
    pub fn iterations(&self) -> usize {
        self.fit.iterations()
    }
    /// Averaged squared loss plus the complete optimization-scale penalty.
    pub fn objective(&self) -> f64 {
        self.fit.objective()
    }
    /// Selected full-space convergence criterion.
    pub fn stopping_criterion(&self) -> StoppingCriterion {
        self.fit.stopping_criterion()
    }
    /// Full mapping infinity norm, including the free-intercept derivative,
    /// evaluated at the returned parameters using [`Self::step_size`].
    pub fn stopping_value(&self) -> f64 {
        self.fit.stopping_value()
    }
    /// Absolute mapping threshold used to certify convergence.
    pub fn stopping_threshold(&self) -> f64 {
        self.fit.stopping_threshold()
    }
    /// Reciprocal of the final curvature, also used by final diagnostics.
    pub fn step_size(&self) -> f64 {
        self.fit.step_size()
    }
    /// Curvature policy selected for this fit.
    pub fn step_policy(&self) -> HybridStep {
        self.step_policy
    }
    /// Number of committed full PG transitions, excluding diagnostic proxes.
    pub fn pg_steps(&self) -> usize {
        self.pg_steps
    }
    /// Number of complete cluster-coordinate sweeps.
    pub fn coordinate_sweeps(&self) -> usize {
        self.coordinate_sweeps
    }
    /// Number of independently refreshed full-space diagnostic checks.
    pub fn diagnostic_checks(&self) -> usize {
        self.diagnostic_checks
    }
    /// Number of rejected trials followed by a curvature doubling.
    pub fn backtracks(&self) -> usize {
        self.backtracks
    }
    /// Cluster label of each feature, indexing [`Self::cluster_values`].
    pub fn cluster_labels(&self) -> &[usize] {
        &self.labels
    }
    /// Signed optimization-scale values, with positive zeros.
    /// Clustered lasso returns strictly increasing values. Chain fused lasso
    /// returns segment values in feature order, which may repeat. Equal
    /// original-scale values need not share a label when training scales differ.
    pub fn cluster_values(&self) -> &[f64] {
        &self.values
    }

    /// Predict with original-scale parameters without fitting preprocessing.
    ///
    /// # Errors
    /// Reject mismatched dimensions, invalid entries, or nonfinite arithmetic.
    pub fn predict<M: RawColumns<f64> + ?Sized>(&self, x: &M) -> Result<Vec<f64>, FitError> {
        self.fit.predict(x)
    }

    pub(crate) fn from_solution<E>(
        solution: crate::solver::hybrid::Solution,
        preprocessing: Preprocessing,
    ) -> Result<Self, FitError<E>> {
        Ok(Self {
            fit: ProximalFit::from_solution(solution.proximal, preprocessing)?,
            labels: solution.labels,
            values: solution.values,
            pg_steps: solution.pg_steps,
            coordinate_sweeps: solution.coordinate_sweeps,
            diagnostic_checks: solution.diagnostic_checks,
            backtracks: solution.backtracks,
            step_policy: solution.step_policy,
        })
    }
}
