use lazymatrix::RawColumns;

use crate::{FitError, Preprocessing, StoppingCriterion, Termination};

/// Original-scale parameters and full-loss KKT diagnostics from [`crate::Irls`].
///
/// Predictions are linear predictors, including logits for binomial models.
/// Objectives and KKT violations use optimization-scale coefficients. A finite
/// KKT tolerance certifies approximate stationarity, not existence of a finite
/// exact optimum for a separated or constant-boundary binomial model.
#[derive(Clone, Debug)]
#[must_use]
pub struct IrlsFit {
    coefficients: Vec<f64>,
    intercept: f64,
    preprocessing: Preprocessing,
    solution: Diagnostics,
}

#[derive(Clone, Debug)]
struct Diagnostics {
    working_set: crate::WorkingSetDiagnostics,

    iterations: usize,
    coordinate_sweeps: usize,
    objective: f64,
    kkt_violation: f64,
    threshold: f64,
    termination: Termination,
}

impl IrlsFit {
    /// Work performed by heuristic feature selection, separate from safe screening.
    pub fn working_set_diagnostics(&self) -> &crate::WorkingSetDiagnostics {
        &self.solution.working_set
    }

    /// Original-scale coefficients in training-column order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    /// Original-scale intercept, including the correction for centering.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    /// Training normalization reused by this fit.
    pub fn preprocessing(&self) -> &Preprocessing {
        &self.preprocessing
    }
    /// Whether a full KKT check passed or the outer iteration budget expired.
    pub fn termination(&self) -> Termination {
        self.solution.termination
    }
    /// Number of accepted outer IRLS steps.
    pub fn iterations(&self) -> usize {
        self.solution.iterations
    }
    /// Total coordinate sweeps across all quadratic models for this fit.
    pub fn coordinate_sweeps(&self) -> usize {
        self.solution.coordinate_sweeps
    }
    /// Original normalized loss plus the optimization-scale penalty.
    pub fn objective(&self) -> f64 {
        self.solution.objective
    }
    /// Maximum full-loss KKT violation, including the unpenalized intercept.
    pub fn kkt_violation(&self) -> f64 {
        self.solution.kkt_violation
    }
    /// The absolute KKT stopping criterion used for this fit.
    pub fn stopping_criterion(&self) -> StoppingCriterion {
        StoppingCriterion::kkt_violation(self.solution.threshold)
    }
    /// Final full-loss KKT violation.
    pub fn stopping_value(&self) -> f64 {
        self.solution.kkt_violation
    }
    /// Absolute KKT threshold.
    pub fn stopping_threshold(&self) -> f64 {
        self.solution.threshold
    }

    /// Predict linear predictors using original-scale parameters and raw columns.
    ///
    /// # Errors
    /// Reject feature-count mismatches, invalid entries, or nonfinite arithmetic.
    pub fn predict<M: RawColumns<f64> + ?Sized>(&self, x: &M) -> Result<Vec<f64>, FitError> {
        super::predict(&self.coefficients, self.intercept, self.iterations(), x)
    }

    pub(crate) fn from_solution<E>(
        solution: crate::solver::irls::Solution,
        preprocessing: Preprocessing,
    ) -> Result<Self, FitError<E>> {
        let (coefficients, intercept) = super::back_transform(
            solution.coefficients,
            solution.intercept,
            &preprocessing,
            solution.iterations,
        )?;
        Ok(Self {
            coefficients,
            intercept,
            preprocessing,
            solution: Diagnostics {
                working_set: solution.working_set,
                iterations: solution.iterations,
                coordinate_sweeps: solution.coordinate_sweeps,
                objective: solution.objective,
                kkt_violation: solution.kkt_violation,
                threshold: solution.threshold,
                termination: solution.termination,
            },
        })
    }
}
