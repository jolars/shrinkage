use crate::error::{finite, invalid};
use crate::{ElasticNet, FitError, L1, Penalty, Ridge};

/// A convex separable penalty with exact scalar updates and KKT checks.
///
/// The complete value must equal the sum of its coordinate terms. Coupled
/// penalties, such as SLOPE and group lasso, do not satisfy this contract.
pub trait CoordinatePenalty: Penalty<Error = FitError> {
    /// Minimize `curvature * value² / 2 - linear * value + penalty_j(value)`.
    /// Curvature must be finite and nonnegative. At zero curvature, return a
    /// finite minimizer when one exists, or an error for an unbounded subproblem.
    ///
    /// # Errors
    /// Reject invalid parameters, indices, inputs, or nonfinite arithmetic.
    fn minimize_coordinate(
        &self,
        coordinate: usize,
        linear: f64,
        curvature: f64,
    ) -> Result<f64, FitError>;

    /// Return the distance of zero from `loss_gradient + subgradient_j(value)`.
    /// The result must be finite and nonnegative, in gradient units.
    ///
    /// # Errors
    /// Reject invalid parameters, indices, inputs, or nonfinite arithmetic.
    fn coordinate_violation(
        &self,
        coordinate: usize,
        value: f64,
        loss_gradient: f64,
    ) -> Result<f64, FitError>;
}

fn minimize(linear: f64, curvature: f64, l1: f64, l2: f64) -> Result<f64, FitError> {
    if !linear.is_finite() || !curvature.is_finite() || curvature < 0.0 {
        return Err(invalid(
            "coordinate inputs must be finite with nonnegative curvature",
        ));
    }
    let numerator = linear.signum() * (linear.abs() - l1).max(0.0);
    let denominator = finite(curvature + l2, "computing coordinate curvature", 0)?;
    if denominator == 0.0 {
        if numerator == 0.0 {
            return Ok(0.0);
        }
        return Err(invalid(
            "zero-curvature coordinate subproblem has no finite minimizer",
        ));
    }
    Ok(finite(
        numerator / denominator,
        "minimizing a coordinate",
        0,
    )?)
}

fn violation(value: f64, gradient: f64, l1: f64, l2: f64) -> Result<f64, FitError> {
    if !value.is_finite() || !gradient.is_finite() {
        return Err(invalid("coordinate values and gradients must be finite"));
    }
    let smooth = finite(
        gradient + l2 * value,
        "computing coordinate stationarity",
        0,
    )?;
    Ok(finite(
        if value == 0.0 {
            (smooth.abs() - l1).max(0.0)
        } else {
            (smooth + l1 * value.signum()).abs()
        },
        "checking coordinate stationarity",
        0,
    )?)
}

impl CoordinatePenalty for L1 {
    fn minimize_coordinate(&self, _: usize, linear: f64, curvature: f64) -> Result<f64, FitError> {
        self.validate()?;
        minimize(linear, curvature, self.lambda(), 0.0)
    }
    fn coordinate_violation(&self, _: usize, value: f64, gradient: f64) -> Result<f64, FitError> {
        self.validate()?;
        violation(value, gradient, self.lambda(), 0.0)
    }
}

impl CoordinatePenalty for Ridge {
    fn minimize_coordinate(&self, _: usize, linear: f64, curvature: f64) -> Result<f64, FitError> {
        L1::new(self.lambda()).validate()?;
        minimize(linear, curvature, 0.0, self.lambda())
    }
    fn coordinate_violation(&self, _: usize, value: f64, gradient: f64) -> Result<f64, FitError> {
        L1::new(self.lambda()).validate()?;
        violation(value, gradient, 0.0, self.lambda())
    }
}

impl CoordinatePenalty for ElasticNet {
    fn minimize_coordinate(&self, _: usize, linear: f64, curvature: f64) -> Result<f64, FitError> {
        self.validate()?;
        minimize(linear, curvature, self.l1(), self.l2())
    }
    fn coordinate_violation(&self, _: usize, value: f64, gradient: f64) -> Result<f64, FitError> {
        self.validate()?;
        violation(value, gradient, self.l1(), self.l2())
    }
}
