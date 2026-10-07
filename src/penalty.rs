//! Penalty values, complete proximal maps, and coefficient subproblems.

use crate::FitError;
use crate::error::{finite, invalid};

/// An L1 penalty on optimization-scale coefficients.
#[derive(Clone, Copy, Debug)]
pub struct L1 {
    pub(crate) lambda: f64,
}

impl L1 {
    /// Set the penalty strength, checked when fitting.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

/// The value of a complete penalty on optimization-scale coefficients.
///
/// A value alone does not imply a proximal map or a valid coordinate update.
pub trait Penalty {
    /// Failure returned by penalty evaluation.
    type Error;

    /// Evaluate the penalty, including its strength.
    ///
    /// # Errors
    /// Returns an error for invalid coefficients, parameters, or arithmetic.
    fn value(&self, coefficients: &[f64]) -> Result<f64, Self::Error>;
}

impl L1 {
    /// Configured penalty strength.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub(crate) fn validate<E>(&self) -> Result<(), FitError<E>> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(invalid("lambda must be finite and nonnegative"));
        }
        Ok(())
    }

    pub(crate) fn value_unchecked(&self, coefficients: &[f64]) -> f64 {
        if self.lambda == 0.0 {
            0.0
        } else {
            coefficients
                .iter()
                .map(|value| self.lambda * value.abs())
                .sum()
        }
    }

    pub(crate) fn coordinate_minimum(&self, correlation: f64, curvature: f64) -> f64 {
        let thresholded = if correlation > self.lambda {
            correlation - self.lambda
        } else if correlation < -self.lambda {
            correlation + self.lambda
        } else {
            0.0
        };
        thresholded / curvature
    }
}

impl Penalty for L1 {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, Self::Error> {
        self.validate()?;
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(invalid("coefficients must be finite"));
        }
        Ok(finite(
            self.value_unchecked(coefficients),
            "computing the penalty",
            0,
        )?)
    }
}

/// A convex penalty with an exact proximal map for the complete term.
///
/// Implementations minimize `||z - input||² / (2 * step) + self.value(z)`.
/// This contract applies to the entire penalty, including its strengths. A
/// sum of penalties does not acquire this capability from its summands.
/// A value-only penalty cannot be fitted by proximal gradient:
///
/// ```compile_fail
/// use shrinkage::{FitError, Gaussian, Penalty, Problem, ProximalGradient};
/// use shrinkage::lazymatrix::{ColumnStats, MatVecInto, MatTransposeVecInto, RawColumns};
/// fn fit<M, P>(problem: &Problem<&M, Gaussian<'_>, P>)
/// where
///     M: RawColumns<f64> + ColumnStats<f64>
///         + MatVecInto<Vec<f64>> + MatTransposeVecInto<Vec<f64>>,
///     P: Penalty<Error = FitError>,
/// {
///     let _ = problem.fit_with(&ProximalGradient::<Vec<f64>>::new());
/// }
/// ```
pub trait ProximalPenalty: Penalty<Error = FitError> {
    /// Write the proximal minimizer into reusable output storage.
    ///
    /// # Errors
    /// Reject invalid parameters, nonpositive or nonfinite steps, mismatched
    /// lengths, nonfinite input, or nonfinite arithmetic. Output is unspecified
    /// after an error.
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError>;
}

/// Ridge penalty `lambda * ||theta||² / 2`, distinct from the L2 norm.
#[derive(Clone, Copy, Debug)]
pub struct Ridge {
    lambda: f64,
}

impl Ridge {
    /// Set the squared L2 strength, checked during evaluation and fitting.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Configured squared L2 strength.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// Complete elastic-net penalty `l1 * ||theta||₁ + l2 * ||theta||² / 2`.
#[derive(Clone, Copy, Debug)]
pub struct ElasticNet {
    l1: f64,
    l2: f64,
}

impl ElasticNet {
    /// Set the L1 and squared L2 strengths independently, checked when used.
    pub fn new(l1: f64, l2: f64) -> Self {
        Self { l1, l2 }
    }

    /// Configured L1 strength.
    pub fn l1(&self) -> f64 {
        self.l1
    }

    /// Configured squared L2 strength.
    pub fn l2(&self) -> f64 {
        self.l2
    }
}

impl Penalty for Ridge {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        ElasticNet::new(0.0, self.lambda).value(coefficients)
    }
}

impl Penalty for ElasticNet {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        L1::new(self.l1).validate()?;
        L1::new(self.l2).validate()?;
        let l1 = L1::new(self.l1).value(coefficients)?;
        let l2 = if self.l2 == 0.0 {
            0.0
        } else {
            coefficients
                .iter()
                .map(|&coefficient| {
                    let scaled = self.l2.sqrt() * coefficient;
                    0.5 * scaled * scaled
                })
                .sum()
        };
        Ok(finite(l1 + l2, "computing the penalty", 0)?)
    }
}

impl ProximalPenalty for L1 {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        ElasticNet::new(self.lambda, 0.0).prox(input, step, output)
    }
}

impl ProximalPenalty for Ridge {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        ElasticNet::new(0.0, self.lambda).prox(input, step, output)
    }
}

impl ProximalPenalty for ElasticNet {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        L1::new(self.l1).validate()?;
        L1::new(self.l2).validate()?;
        if input.len() != output.len() {
            return Err(invalid("proximal input and output lengths must match"));
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid(
                "proximal step must be finite and strictly positive",
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(invalid("proximal input must be finite"));
        }
        let threshold = finite(step * self.l1, "computing a proximal threshold", 0)?;
        let denominator = finite(1.0 + step * self.l2, "computing proximal curvature", 0)?;
        for (&value, target) in input.iter().zip(output) {
            *target = value.signum() * (value.abs() - threshold).max(0.0) / denominator;
        }
        Ok(())
    }
}
