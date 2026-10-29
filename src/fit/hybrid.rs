use lazymatrix::RawColumns;

use crate::{FitError, HybridStep, Preprocessing, ProximalFit, StoppingCriterion, Termination};

/// Original-scale parameters and full-space hybrid diagnostics.
///
/// Cluster labels and values refer to optimization-scale coefficients.
/// Clustered lasso orders blocks by value; chain fused lasso orders contiguous
/// segments by feature and retains zero-weight edges as segment boundaries.
/// SLOPE orders equal-magnitude blocks decreasingly and retains individual
/// signs in [`Self::coefficient_signs`].
/// Scaling can change equalities among the reported original-scale parameters.
/// Gap Safe certificates refer to the original loss and full design. A sweep
/// that leaves all blocks unchanged does not certify convergence; check
/// [`Self::termination`] and the full mapping.
#[derive(Clone, Debug)]
#[must_use]
pub struct HybridFit {
    working_set: crate::WorkingSetDiagnostics,

    fit: ProximalFit,
    labels: Vec<usize>,
    values: Vec<f64>,
    signs: Vec<f64>,
    pg_steps: usize,
    coordinate_sweeps: usize,
    diagnostic_checks: usize,
    backtracks: usize,
    step_policy: HybridStep,
    outer_iterations: usize,
    outer_backtracks: usize,
}

impl HybridFit {
    /// Work performed by heuristic feature selection, separate from safe screening.
    pub fn working_set_diagnostics(&self) -> &crate::WorkingSetDiagnostics {
        &self.working_set
    }

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
    /// Total committed PG transitions plus coordinate sweeps over the current working set.
    pub fn iterations(&self) -> usize {
        self.fit.iterations()
    }
    /// Original averaged loss plus the complete optimization-scale penalty.
    pub fn objective(&self) -> f64 {
        self.fit.objective()
    }

    /// Original-space feasible dual vector from the latest certificate check.
    /// Screening-disabled fits do not construct a certificate.
    pub fn dual_certificate(&self) -> Option<&[f64]> {
        self.fit.dual_certificate()
    }

    /// Original-loss dual lower bound, when a certificate is available.
    pub fn dual_objective(&self) -> Option<f64> {
        self.fit.dual_objective()
    }

    /// Original objective minus its feasible dual lower bound.
    /// This diagnostic does not replace the full mapping stopping rule.
    pub fn duality_gap(&self) -> Option<f64> {
        self.fit.duality_gap()
    }

    /// Work and certified zero restrictions in the original feature space.
    pub fn screening_diagnostics(&self) -> &crate::screening::ScreeningDiagnostics {
        self.fit.screening_diagnostics()
    }
    /// Selected full-space convergence criterion.
    pub fn stopping_criterion(&self) -> StoppingCriterion {
        self.fit.stopping_criterion()
    }
    /// Full mapping infinity norm, including the free-intercept derivative,
    /// evaluated for the original loss at the returned parameters using
    /// [`Self::step_size`]. Logistic surrogate checks never certify convergence.
    pub fn stopping_value(&self) -> f64 {
        self.fit.stopping_value()
    }
    /// Absolute mapping threshold used to certify convergence.
    pub fn stopping_threshold(&self) -> f64 {
        self.fit.stopping_threshold()
    }
    /// Step used by final full-space diagnostics. Gaussian fits use the final
    /// PG curvature; logistic fits use a fixed global logistic curvature bound.
    pub fn step_size(&self) -> f64 {
        self.fit.step_size()
    }
    /// Curvature policy selected for this fit.
    pub fn step_policy(&self) -> HybridStep {
        self.step_policy
    }
    /// Number of committed PG transitions, excluding diagnostic proxes.
    pub fn pg_steps(&self) -> usize {
        self.pg_steps
    }
    /// Number of cluster-coordinate sweeps over the current working set.
    pub fn coordinate_sweeps(&self) -> usize {
        self.coordinate_sweeps
    }
    /// Number of refreshed inner diagnostic checks. Logistic checks use the
    /// fixed-zero quadratic subproblem; original-loss checks also run initially
    /// and after every accepted outer step.
    pub fn diagnostic_checks(&self) -> usize {
        self.diagnostic_checks
    }
    /// Number of rejected trials followed by a curvature doubling.
    pub fn backtracks(&self) -> usize {
        self.backtracks
    }
    /// Accepted logistic IRLS steps. Zero for Gaussian fits.
    pub fn outer_iterations(&self) -> usize {
        self.outer_iterations
    }
    /// Rejected logistic trials followed by a step halving. Zero for Gaussian.
    pub fn outer_backtracks(&self) -> usize {
        self.outer_backtracks
    }
    /// Cluster label of each feature, indexing [`Self::cluster_values`].
    pub fn cluster_labels(&self) -> &[usize] {
        &self.labels
    }
    /// Optimization-scale block values, with positive zeros.
    /// Clustered lasso returns strictly increasing values. Chain fused lasso
    /// returns signed segment values in feature order, which may repeat. SLOPE
    /// returns strictly decreasing nonnegative magnitudes, with zero last when
    /// present. For SLOPE, coefficient `j` on the optimization scale equals
    /// `coefficient_signs()[j] * cluster_values()[cluster_labels()[j]]`. Equal
    /// original-scale values need not share a label when training scales differ.
    pub fn cluster_values(&self) -> &[f64] {
        &self.values
    }

    /// Optimization-scale coefficient signs in feature order: -1, 0, or 1.
    /// Both signed zeros have sign zero. Signs are recorded before coefficient
    /// back-transformation; SLOPE clusters may contain both nonzero signs.
    pub fn coefficient_signs(&self) -> &[f64] {
        &self.signs
    }

    /// Predict linear predictors with original-scale parameters without fitting
    /// preprocessing. Binomial predictions are logits.
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
        let signs = solution
            .proximal
            .coefficients
            .iter()
            .map(|&value| if value == 0.0 { 0.0 } else { value.signum() })
            .collect();
        Ok(Self {
            working_set: solution.working_set,
            fit: ProximalFit::from_solution(solution.proximal, preprocessing)?,
            labels: solution.labels,
            values: solution.values,
            signs,
            pg_steps: solution.pg_steps,
            coordinate_sweeps: solution.coordinate_sweeps,
            diagnostic_checks: solution.diagnostic_checks,
            backtracks: solution.backtracks,
            step_policy: solution.step_policy,
            outer_iterations: solution.outer_iterations,
            outer_backtracks: solution.outer_backtracks,
        })
    }
}
