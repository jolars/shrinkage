//! Solver capabilities, configuration, and stopping rules.

pub(crate) mod coordinate;

use crate::error::invalid;
use crate::{FitError, L1};

/// A solver for a supported problem type.
///
/// Implementations own their compatibility bounds and result and error types.
/// Unsupported combinations have no implementation.
pub trait Solver<P> {
    /// Owned result returned after fitting.
    type Fit;
    /// Input, numerical, or backend failure.
    type Error;

    /// Fit a problem using this solver.
    ///
    /// # Errors
    /// Returns an error if validation or a required numerical operation fails.
    fn solve(&self, problem: &P) -> Result<Self::Fit, Self::Error>;
}

/// Cyclic coordinate descent settings for Gaussian lasso.
#[derive(Clone, Copy, Debug)]
pub struct CoordinateDescent {
    stopping_criterion: Option<StoppingCriterion>,
    max_iterations: usize,
}

impl CoordinateDescent {
    /// Use a relative duality-gap tolerance of `1e-6` and at most 10,000 sweeps.
    /// Zero-penalty problems use the lasso's absolute KKT default instead.
    pub fn new() -> Self {
        Self {
            stopping_criterion: None,
            max_iterations: 10_000,
        }
    }

    /// Set the absolute KKT tolerance, checked when fitting.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.stopping_criterion = Some(StoppingCriterion::kkt_violation(tolerance));
        self
    }

    /// Select a stopping criterion with its own tolerances.
    pub fn terminate_on(mut self, criterion: StoppingCriterion) -> Self {
        self.stopping_criterion = Some(criterion);
        self
    }

    /// Set the maximum number of complete sweeps, checked when fitting.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

impl Default for CoordinateDescent {
    fn default() -> Self {
        Self::new()
    }
}

/// A convergence check with tolerances in its own units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoppingCriterion {
    /// Stop when the absolute Gaussian lasso duality gap is at most
    /// `absolute + relative * reference_loss`.
    DualityGap {
        /// Absolute gap tolerance.
        absolute: f64,
        /// Gap tolerance relative to the fixed zero-coefficient loss.
        relative: f64,
    },
    /// Stop when the maximum absolute KKT violation is at most `absolute`.
    KktViolation {
        /// Absolute KKT tolerance.
        absolute: f64,
    },
}

impl StoppingCriterion {
    /// Select a relative duality-gap tolerance.
    pub const fn duality_gap(relative: f64) -> Self {
        Self::DualityGap {
            absolute: 0.0,
            relative,
        }
    }

    /// Select an absolute KKT tolerance.
    pub const fn kkt_violation(absolute: f64) -> Self {
        Self::KktViolation { absolute }
    }
}

impl CoordinateDescent {
    pub(crate) fn criterion(&self, penalty: L1) -> StoppingCriterion {
        self.stopping_criterion.unwrap_or_else(|| {
            if penalty.lambda == 0.0 {
                StoppingCriterion::kkt_violation(1e-6)
            } else {
                StoppingCriterion::duality_gap(1e-6)
            }
        })
    }

    pub(crate) fn validate<E>(&self, criterion: StoppingCriterion) -> Result<(), FitError<E>> {
        match criterion {
            StoppingCriterion::KktViolation { absolute } => {
                if !absolute.is_finite() || absolute <= 0.0 {
                    return Err(invalid(
                        "KKT tolerance must be finite and strictly positive",
                    ));
                }
            }
            StoppingCriterion::DualityGap { absolute, relative } => {
                if !absolute.is_finite()
                    || !relative.is_finite()
                    || absolute < 0.0
                    || relative < 0.0
                    || (absolute == 0.0 && relative == 0.0)
                {
                    return Err(invalid(
                        "duality-gap tolerances must be finite, nonnegative, and at least one positive",
                    ));
                }
            }
        }
        if self.max_iterations == 0 {
            return Err(invalid("max_iterations must be strictly positive"));
        }
        Ok(())
    }
}
