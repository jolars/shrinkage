//! Owned fitted parameters, training transformations, and diagnostics.

use crate::error::{finite, invalid};
use crate::problem::validate_matrix;
use crate::{Centering, LassoError, Scaling, StoppingCriterion};
use lazymatrix::{RawColumn, RawColumns};

/// Training normalization retained by a fitted model.
#[derive(Clone, Debug)]
pub struct Preprocessing {
    pub(crate) spec: lazymatrix::Normalization,
    pub(crate) centers: Option<Vec<f64>>,
    pub(crate) scales: Option<Vec<f64>>,
    pub(crate) supplied_centers: bool,
    pub(crate) supplied_scales: bool,
}

impl Preprocessing {
    /// The configured centering rule, with automatic choices resolved.
    /// A supplied center vector overrides this rule.
    pub fn centering(&self) -> Centering {
        self.spec.center
    }

    /// The configured scaling rule, with automatic choices resolved.
    /// A supplied scale vector overrides this rule.
    pub fn scaling(&self) -> Scaling {
        self.spec.scale
    }

    /// Training-column centers, or `None` when centering was disabled.
    pub fn centers(&self) -> Option<&[f64]> {
        self.centers.as_deref()
    }

    /// Training-column scales with zeros replaced by one, or `None` when disabled.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Whether the centers were supplied rather than computed during fitting.
    pub fn centers_were_supplied(&self) -> bool {
        self.supplied_centers
    }

    /// Whether the scales were supplied rather than computed during fitting.
    pub fn scales_were_supplied(&self) -> bool {
        self.supplied_scales
    }
}

/// Why a finite fit stopped. An iteration limit does not certify optimality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The selected check passed using freshly evaluated model state.
    Converged,
    /// The iteration budget was exhausted before meeting the selected threshold.
    IterationLimit,
}

/// Owned original-scale parameters, preprocessing metadata, and diagnostics.
///
/// Check [`Self::termination`] before treating a fit as converged. Diagnostics
/// refer to the optimization-scale penalty even though coefficients use the
/// original scale.
#[derive(Clone, Debug)]
#[must_use]
pub struct LassoFit {
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

    /// Number of complete coordinate sweeps performed.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Final `||residual||² / (2n) + lambda * ||theta||₁`.
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

pub(crate) fn back_transform<E>(
    mut coefficients: Vec<f64>,
    mut intercept: f64,
    preprocessing: &Preprocessing,
    iterations: usize,
) -> Result<(Vec<f64>, f64), LassoError<E>> {
    if let Some(scales) = &preprocessing.scales {
        for (coefficient, scale) in coefficients.iter_mut().zip(scales) {
            *coefficient = finite(
                *coefficient / scale,
                "back-transforming coefficients",
                iterations,
            )?;
        }
    }
    if let Some(centers) = &preprocessing.centers {
        for (coefficient, center) in coefficients.iter().zip(centers) {
            intercept = finite(
                intercept - coefficient * center,
                "back-transforming intercept",
                iterations,
            )?;
        }
    }
    Ok((coefficients, intercept))
}

fn predict<M: RawColumns<f64> + ?Sized>(
    coefficients: &[f64],
    intercept: f64,
    iterations: usize,
    x: &M,
) -> Result<Vec<f64>, LassoError> {
    if x.ncols() != coefficients.len() {
        return Err(invalid(format!(
            "prediction design has {} columns; expected {}",
            x.ncols(),
            coefficients.len()
        )));
    }
    validate_matrix(x)?;
    let mut predictions = vec![intercept; x.nrows()];
    for (j, &coefficient) in coefficients.iter().enumerate() {
        if coefficient != 0.0 {
            x.raw_column(j)
                .affine_add_to(coefficient, 0.0, &mut predictions);
        }
    }
    for &value in &predictions {
        finite(value, "predicting", iterations)?;
    }
    Ok(predictions)
}

/// Owned original-scale parameters and proximal-gradient diagnostics.
///
/// The objective and mapping use optimization-scale coefficients. No duality
/// gap is reported because this solver has no implemented dual certificate.
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
}

impl ProximalFit {
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
    /// Infinity norm of `(theta - prox(theta - step * gradient)) / step`,
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
        })
    }
}
