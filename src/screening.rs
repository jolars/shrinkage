//! Feasible dual certificates and dynamic, fixed-zero restrictions.

pub(crate) mod intercept;
mod penalty;

use crate::error::finite;
use crate::{FitError, ProximalDesign, ProximalPenalty, SmoothDatafit};

/// Original-problem certificate on the observation scale.
#[derive(Clone, Debug)]
pub struct DualCertificate {
    vector: Vec<f64>,
    objective: f64,
    gap: f64,
    radius: f64,
}

impl DualCertificate {
    /// Feasible observation-scale dual vector `theta`, in training-row order.
    pub fn dual_vector(&self) -> &[f64] {
        &self.vector
    }
    /// Dual objective `D(theta)` for the original, complete problem.
    pub fn dual_objective(&self) -> f64 {
        self.objective
    }
    /// Conservatively padded `G(beta_0, beta, theta)` on the averaged-loss scale.
    pub fn duality_gap(&self) -> f64 {
        self.gap
    }
    /// Euclidean radius containing the optimal observation-scale dual vector.
    pub fn radius(&self) -> f64 {
        self.radius
    }
}

/// Work performed by safe screening, rather than sparsity of the iterate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScreeningDiagnostics {
    /// Whether all components supplied the necessary capabilities.
    pub supported: bool,
    /// Number of original-loss certificate constructions.
    pub certificate_passes: usize,
    /// Number of unresolved coefficients tested for zero screening.
    pub zero_tests: usize,
    /// Number of coefficients fixed to zero by a certificate.
    pub screened_features: usize,
    /// Number of original features still eligible for numerical updates.
    pub active_dimension: usize,
    /// Number of changes to the fixed-zero restriction.
    pub rebuilds: usize,
    /// Number of universally feasible zero certificates used as fallbacks.
    pub fallback_certificates: usize,
}

/// Loss-side capability for original-loss Gap Safe certificates.
pub trait GapSafeDatafit {
    /// Construct a domain-feasible candidate, including any intercept equality.
    ///
    /// # Errors
    /// Reject invalid dimensions, inputs, or nonfinite arithmetic.
    fn dual_candidate(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError>;
    /// Repair domain and equality constraints after floating-point scaling.
    ///
    /// # Errors
    /// Reject invalid dimensions or arithmetic. A feasible zero fallback is valid.
    fn finalize_dual(&self, dual: &mut [f64]) -> Result<(), FitError>;
    /// Evaluate the dual objective, excluding penalty feasibility.
    ///
    /// # Errors
    /// Reject a candidate outside the loss conjugate's domain.
    fn dual_objective(&self, dual: &[f64]) -> Result<f64, FitError>;
    /// Evaluate the loss Fenchel gap without subtracting full objectives.
    ///
    /// This is `F(eta) + F_conjugate(-theta) + dot(eta, theta)`.
    ///
    /// # Errors
    /// Reject incompatible dimensions, invalid domains, or nonfinite arithmetic.
    fn fenchel_gap(&self, predictor: &[f64], dual: &[f64]) -> Result<f64, FitError>;
    /// Positive strong-concavity constant on the observation scale.
    fn strong_concavity(&self) -> f64;
}

/// Complete-penalty dual feasibility and exact fixed-zero restrictions.
///
/// A fixed-zero constraint differs from excluding a coordinate from a penalty.
/// Implementations must retain every interaction with removed zeros.
pub trait GapSafePenalty: ProximalPenalty {
    /// Whether this penalty has a positive sparsity component.
    fn has_sparsity(&self) -> bool;
    /// Check complete feasibility for all correlations within the error bounds.
    ///
    /// # Errors
    /// Reject incompatible dimensions or invalid penalty settings.
    fn dual_feasible(&self, correlations: &[f64], errors: &[f64]) -> Result<bool, FitError>;
    /// Return an analytic feasible divisor, or request the generic search.
    ///
    /// # Errors
    /// Reject incompatible dimensions or invalid penalty settings.
    fn dual_scaling(
        &self,
        _correlations: &[f64],
        _errors: &[f64],
    ) -> Result<Option<f64>, FitError> {
        Ok(None)
    }
    /// Add proven zeros using bounds on the optimal correlations.
    ///
    /// `errors[j]` includes the safe radius times the column norm and rounding.
    /// Existing `true` entries must remain fixed.
    ///
    /// # Errors
    /// Reject incompatible dimensions or invalid penalty settings.
    fn screen_zeros(
        &self,
        correlations: &[f64],
        errors: &[f64],
        zeroed: &mut [bool],
    ) -> Result<(), FitError>;
    /// Evaluate the exact prox with `true` coordinates constrained to zero.
    ///
    /// # Errors
    /// Reject invalid parameters, dimensions, steps, or arithmetic.
    fn prox_zeroed(
        &self,
        input: &[f64],
        zeroed: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError>;
}

/// Optional normalized-column operations that avoid screened matrix work.
pub trait ScreeningDesign {
    /// Backend failure type.
    type Error;
    /// Write upper bounds for Euclidean norms of optimization-scale columns.
    ///
    /// # Errors
    /// Reject incompatible output dimensions or nonfinite arithmetic.
    fn column_norms(&mut self, output: &mut [f64]) -> Result<(), FitError<Self::Error>>;
    /// Write absolute product scales accounting for normalization cancellation.
    ///
    /// A scale must bound the magnitudes entering a transposed product before
    /// cancellation, per unit Euclidean input norm.
    ///
    /// # Errors
    /// Reject incompatible dimensions or unrepresentable scales.
    fn column_error_scales(&mut self, output: &mut [f64]) -> Result<(), FitError<Self::Error>>;
    /// Multiply using only increasing, unique original indices in `active`.
    ///
    /// Input has the original coefficient dimension; output has one entry per row.
    ///
    /// # Errors
    /// Reject invalid dimensions or indices and preserve backend failures.
    fn forward_active(
        &mut self,
        input: &[f64],
        active: &[usize],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>>;
    /// Write selected correlations into an original-length output, zeroing others.
    ///
    /// `active` contains increasing, unique original feature indices.
    ///
    /// # Errors
    /// Reject invalid dimensions or indices and preserve backend failures.
    fn transpose_active(
        &mut self,
        input: &[f64],
        active: &[usize],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>>;
}

pub(crate) struct ScreeningWorkspace {
    pub(crate) zeroed: Vec<bool>,
    pub(crate) active: Vec<usize>,
    norms: Vec<f64>,
    error_scales: Vec<f64>,
    dual: Vec<f64>,
    correlations: Vec<f64>,
    errors: Vec<f64>,
    trial: Vec<f64>,
    trial_errors: Vec<f64>,
    candidates: Vec<bool>,
    certificate: Option<DualCertificate>,
    diagnostics: ScreeningDiagnostics,
}

impl ScreeningWorkspace {
    pub(crate) fn new(_n: usize, p: usize) -> Self {
        Self {
            zeroed: vec![false; p],
            active: (0..p).collect(),
            norms: Vec::new(),
            error_scales: Vec::new(),
            dual: Vec::new(),
            correlations: Vec::new(),
            errors: Vec::new(),
            trial: Vec::new(),
            trial_errors: Vec::new(),
            candidates: Vec::new(),
            certificate: None,
            diagnostics: ScreeningDiagnostics {
                active_dimension: p,
                ..Default::default()
            },
        }
    }

    pub(crate) fn reset(&mut self) {
        self.zeroed.fill(false);
        self.active.clear();
        self.active.extend(0..self.zeroed.len());
        self.certificate = None;
        self.diagnostics = ScreeningDiagnostics {
            active_dimension: self.zeroed.len(),
            ..Default::default()
        };
    }

    pub(crate) fn diagnostics(&self) -> ScreeningDiagnostics {
        self.diagnostics.clone()
    }
    pub(crate) fn certificate_owned(&self) -> Option<DualCertificate> {
        self.certificate.clone()
    }

    pub(crate) fn forward<O: ProximalDesign + ?Sized>(
        &self,
        design: &mut O,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FitError<O::Error>> {
        if self.diagnostics.screened_features > 0 {
            if let Some(columns) = design.screening_design() {
                return columns.forward_active(input, &self.active, output);
            }
        }
        design.forward(input, output)
    }

    pub(crate) fn transpose<O: ProximalDesign + ?Sized>(
        &self,
        design: &mut O,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FitError<O::Error>> {
        if self.diagnostics.screened_features > 0 {
            if let Some(columns) = design.screening_design() {
                return columns.transpose_active(input, &self.active, output);
            }
        }
        design.transpose(input, output)
    }

    pub(crate) fn prox<P: ProximalPenalty + ?Sized>(
        &self,
        penalty: &P,
        input: &[f64],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        if self.diagnostics.screened_features > 0 {
            if let Some(capability) = penalty.gap_safe() {
                return capability.prox_zeroed(input, &self.zeroed, step, output);
            }
        }
        penalty.prox(input, step, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn check<O, D, P>(
        &mut self,
        design: &mut O,
        datafit: &D,
        penalty: &P,
        coefficients: &[f64],
        predictor: &[f64],
        objective: f64,
        iteration: usize,
        allow_reduction: bool,
    ) -> Result<bool, FitError<O::Error>>
    where
        O: ProximalDesign + ?Sized,
        D: SmoothDatafit + ?Sized,
        P: ProximalPenalty + ?Sized,
    {
        let (Some(loss), Some(term)) = (datafit.gap_safe(), penalty.gap_safe()) else {
            return Ok(false);
        };
        let Some(columns) = design.screening_design() else {
            return Ok(false);
        };
        let n = predictor.len();
        let p = coefficients.len();
        let mut fallback = !term.has_sparsity();
        if !fallback && self.norms.len() != p {
            self.norms.resize(p, 0.0);
            self.error_scales.resize(p, 0.0);
            let bounds = columns
                .column_norms(&mut self.norms)
                .and_then(|()| columns.column_error_scales(&mut self.error_scales));
            match bounds {
                Ok(()) => (),
                Err(FitError::NumericalFailure { .. }) => {
                    self.norms.clear();
                    self.error_scales.clear();
                    fallback = true;
                }
                Err(error) => return Err(error),
            }
            let factor = 1.0 + 64.0 * f64::EPSILON * (n + p + 1) as f64;
            for value in &mut self.norms {
                *value *= factor;
            }
        }
        self.diagnostics.supported = true;
        self.diagnostics.certificate_passes += 1;
        self.dual.resize(n, 0.0);
        let rounding = 128.0 * f64::EPSILON * (n + p + 1) as f64;
        if !fallback {
            self.correlations.resize(p, 0.0);
            self.errors.resize(p, 0.0);
            self.trial.resize(p, 0.0);
            self.trial_errors.resize(p, 0.0);
            self.candidates.resize(p, false);
            loss.dual_candidate(predictor, &mut self.dual)
                .map_err(|e| FitError::from_component(e, iteration))?;
            design.transpose(&self.dual, &mut self.correlations)?;
            let norm = self.dual.iter().fold(0.0_f64, |a, &b| a.hypot(b));
            for (e, &scale) in self.errors.iter_mut().zip(&self.error_scales) {
                *e = product_error(rounding, scale, norm);
            }
            let analytic = term
                .dual_scaling(&self.correlations, &self.errors)
                .map_err(|e| FitError::from_component(e, iteration))?;
            let mut upper = analytic.unwrap_or(1.0).max(1.0);
            fallback = !upper.is_finite();
            let feasible = |divisor: f64,
                            scratch: &mut [f64],
                            scaled_errors: &mut [f64],
                            errors: &[f64]|
             -> Result<bool, FitError> {
                for (((target, scaled), &g), &e) in scratch
                    .iter_mut()
                    .zip(scaled_errors.iter_mut())
                    .zip(&self.correlations)
                    .zip(errors)
                {
                    *target = g / divisor;
                    *scaled = e / divisor;
                    if !target.is_finite() || !e.is_finite() {
                        return Ok(false);
                    }
                }
                term.dual_feasible(scratch, scaled_errors)
            };
            if !fallback
                && !feasible(upper, &mut self.trial, &mut self.trial_errors, &self.errors)
                    .map_err(|e| FitError::from_component(e, iteration))?
            {
                while upper.is_finite()
                    && !feasible(upper, &mut self.trial, &mut self.trial_errors, &self.errors)
                        .map_err(|e| FitError::from_component(e, iteration))?
                {
                    upper *= 2.0;
                }
                if upper.is_finite() {
                    let mut lower = upper / 2.0;
                    for _ in 0..if analytic.is_none() { 64 } else { 0 } {
                        let middle = lower + (upper - lower) / 2.0;
                        if middle == lower || middle == upper {
                            break;
                        }
                        if feasible(
                            middle,
                            &mut self.trial,
                            &mut self.trial_errors,
                            &self.errors,
                        )
                        .map_err(|e| FitError::from_component(e, iteration))?
                        {
                            upper = middle;
                        } else {
                            lower = middle;
                        }
                    }
                } else {
                    fallback = true;
                }
            }
            if !fallback {
                upper *= 1.0 + rounding;
                for theta in &mut self.dual {
                    *theta /= upper;
                }
                loss.finalize_dual(&mut self.dual)
                    .map_err(|e| FitError::from_component(e, iteration))?;
                design.transpose(&self.dual, &mut self.correlations)?;
                let norm = self.dual.iter().fold(0.0_f64, |a, &b| a.hypot(b));
                for (e, &scale) in self.errors.iter_mut().zip(&self.error_scales) {
                    *e = product_error(rounding, scale, norm);
                }
                fallback = !term
                    .dual_feasible(&self.correlations, &self.errors)
                    .map_err(|e| FitError::from_component(e, iteration))?;
            }
        }
        if fallback {
            self.dual.fill(0.0);
            self.correlations.fill(0.0);
            self.errors.fill(0.0);
            self.diagnostics.fallback_certificates += 1;
        }
        let dual_objective = loss
            .dual_objective(&self.dual)
            .map_err(|e| FitError::from_component(e, iteration))?;
        let penalty_value = penalty
            .value(coefficients)
            .map_err(|e| FitError::from_component(e, iteration))?;
        let support: f64 = coefficients
            .iter()
            .zip(&self.correlations)
            .map(|(&b, &g)| b * g)
            .sum();
        let support_magnitude: f64 = coefficients
            .iter()
            .zip(&self.correlations)
            .map(|(&b, &g)| b.abs() * g.abs())
            .sum();
        let support_error: f64 = coefficients
            .iter()
            .zip(&self.errors)
            .map(|(&b, &e)| b.abs() * e)
            .sum();
        let loss_gap = loss
            .fenchel_gap(predictor, &self.dual)
            .map_err(|e| FitError::from_component(e, iteration))?;
        let predictor_norm = predictor.iter().fold(0.0_f64, |a, &b| a.hypot(b));
        let dual_norm = self.dual.iter().fold(0.0_f64, |a, &b| a.hypot(b));
        // The predictor term also covers rounding of the free-intercept
        // equality after radial scaling of a centered certificate.
        let guard = support_error
            + rounding
                * (objective.abs()
                    + dual_objective.abs()
                    + penalty_value.abs()
                    + support_magnitude
                    + predictor_norm * dual_norm)
                    .max(f64::MIN_POSITIVE);
        let raw_gap = finite(
            loss_gap + (penalty_value - support),
            "computing a screening duality gap",
            iteration,
        )?;
        if raw_gap < -guard {
            return Err(FitError::NumericalFailure {
                operation: "validating screening weak duality",
                iteration,
            });
        }
        let gap = finite(
            raw_gap.max(0.0) + guard,
            "rounding a screening duality gap",
            iteration,
        )?;
        let radius = finite(
            (2.0 * (gap / loss.strong_concavity())).sqrt(),
            "computing a Gap Safe radius",
            iteration,
        )?;
        let changed = if allow_reduction && term.has_sparsity() && !fallback {
            for (e, &norm) in self.errors.iter_mut().zip(&self.norms) {
                *e += radius * norm;
            }
            self.candidates.copy_from_slice(&self.zeroed);
            term.screen_zeros(&self.correlations, &self.errors, &mut self.candidates)
                .map_err(|e| FitError::from_component(e, iteration))?;
            self.diagnostics.zero_tests += self.active.len();
            let mut changed = false;
            for ((zero, &candidate), &coefficient) in self
                .zeroed
                .iter_mut()
                .zip(&self.candidates)
                .zip(coefficients)
            {
                if candidate && coefficient == 0.0 && !*zero {
                    *zero = true;
                    changed = true;
                }
            }
            if changed {
                self.active.retain(|&j| !self.zeroed[j]);
                self.diagnostics.screened_features = p - self.active.len();
                self.diagnostics.active_dimension = self.active.len();
                self.diagnostics.rebuilds += 1;
            }
            changed
        } else {
            false
        };
        match &mut self.certificate {
            Some(certificate) => {
                certificate.vector.clone_from(&self.dual);
                certificate.objective = dual_objective;
                certificate.gap = gap;
                certificate.radius = radius;
            }
            None => {
                self.certificate = Some(DualCertificate {
                    vector: self.dual.clone(),
                    objective: dual_objective,
                    gap,
                    radius,
                })
            }
        }
        Ok(changed)
    }
}

// Form the magnitude before the small factor to retain subnormal scales.
fn product_error(rounding: f64, scale: f64, norm: f64) -> f64 {
    if scale == 0.0 || norm == 0.0 {
        return 0.0;
    }
    let product = scale * norm;
    let error = if product.is_finite() {
        rounding * product
    } else {
        (rounding * scale) * norm
    };
    if error == f64::INFINITY {
        error
    } else {
        f64::from_bits(error.to_bits() + 1)
    }
}
