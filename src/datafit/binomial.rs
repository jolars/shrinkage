//! Binomial loss with a logit link.

use crate::error::{finite, invalid};
use crate::{Datafit, DiagonalHessianDatafit, FitError, SmoothDatafit};

/// Bernoulli negative log likelihood with a logit link, averaged over rows.
///
/// Responses must be finite and lie in `[0, 1]`; fractional responses represent
/// proportions with equal observation weights. The complete linear predictor
/// includes an optional unpenalized intercept, fitted by default.
///
/// The predictor Hessian is diagonal with entries at most `1 / (4n)`.
/// Consequently, for an optimization-scale design augmented with an intercept
/// column, the coefficient gradient has Lipschitz constant at most
/// `||[X, 1]||² / (4n)`. [`crate::ProximalGradient`] uses backtracking to enforce
/// the corresponding quadratic upper bound without computing this norm.
/// [`crate::Irls`] uses the exact diagonal predictor Hessian to construct
/// weighted quadratic models solved by coordinate descent.
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::{Col, Mat};
/// use shrinkage::{Binomial, Problem, ProximalGradient, Ridge};
/// let x = Mat::from_fn(4, 1, |i, _| i as f64);
/// let y = [0.0, 0.0, 1.0, 1.0];
/// let fit = Problem::new(&x, Binomial::new(&y), Ridge::new(0.1))
///     .fit_with(&ProximalGradient::<Col<f64>>::new())?;
/// // Predictions are logits; apply the logistic function for probabilities.
/// let logits = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Binomial<'a> {
    response: &'a [f64],
    fit_intercept: bool,
}

impl<'a> Binomial<'a> {
    /// Borrow responses, fitting an intercept by default.
    pub fn new(response: &'a [f64]) -> Self {
        Self {
            response,
            fit_intercept: true,
        }
    }

    /// Enable or disable the unpenalized fitted intercept.
    ///
    /// Explicit column centering can still induce an original-scale intercept.
    pub fn fit_intercept(mut self, enabled: bool) -> Self {
        self.fit_intercept = enabled;
        self
    }

    /// Borrow the observed responses.
    pub fn response(&self) -> &[f64] {
        self.response
    }

    /// Whether fitting includes a free, unpenalized intercept.
    pub fn fits_intercept(&self) -> bool {
        self.fit_intercept
    }

    fn validate(&self, predictor: &[f64]) -> Result<(), FitError> {
        if self.response.is_empty() {
            return Err(invalid("training data must have at least one observation"));
        }
        if predictor.len() != self.response.len() {
            return Err(invalid("response and predictor lengths must match"));
        }
        for (i, &target) in self.response.iter().enumerate() {
            if !target.is_finite() || !(0.0..=1.0).contains(&target) {
                return Err(invalid(format!(
                    "binomial response at row {i} must be finite and in [0, 1]"
                )));
            }
        }
        if predictor.iter().any(|value| !value.is_finite()) {
            return Err(invalid("predictor values must be finite"));
        }
        Ok(())
    }
}

impl Datafit for Binomial<'_> {
    type Error = FitError;

    fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
        self.validate(predictor)?;
        let n = predictor.len() as f64;
        let mut loss = 0.0;
        for (&eta, &target) in predictor.iter().zip(self.response) {
            // Avoid subtracting large, nearly equal terms for correctly
            // classified observations, and scale before summing to avoid overflow.
            let linear = if eta >= 0.0 {
                (1.0 - target) * eta
            } else {
                -target * eta
            };
            loss += linear / n + (-eta.abs()).exp().ln_1p() / n;
        }
        Ok(finite(loss, "computing the binomial loss", 0)?)
    }
}

impl SmoothDatafit for Binomial<'_> {
    fn gap_safe(&self) -> Option<&dyn crate::GapSafeDatafit> {
        Some(self)
    }

    fn nobs(&self) -> usize {
        self.response.len()
    }

    fn fits_intercept(&self) -> bool {
        self.fit_intercept
    }

    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.validate(predictor)?;
        if output.len() != predictor.len() {
            return Err(invalid("predictor and gradient lengths must match"));
        }
        for ((&eta, &target), derivative) in predictor.iter().zip(self.response).zip(output) {
            let tail = (-eta.abs()).exp();
            // This form retains the small negative derivative when target is
            // one and a large positive logit makes the probability round to one.
            let residual = if eta >= 0.0 {
                (1.0 - target) - tail / (1.0 + tail)
            } else {
                tail / (1.0 + tail) - target
            };
            *derivative = residual / predictor.len() as f64;
        }
        Ok(())
    }
}

impl crate::GapSafeDatafit for Binomial<'_> {
    fn dual_candidate(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.gradient(predictor, output)?;
        for theta in output.iter_mut() {
            *theta = -*theta;
        }
        if !self.fit_intercept {
            return Ok(());
        }
        let n = output.len() as f64;
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for (&q, &y) in output.iter().zip(self.response) {
            lower = lower.min(q - y / n);
            upper = upper.max(q - (y - 1.0) / n);
        }
        // Project onto the entropy box and intercept hyperplane together.
        // Centering alone can put binary observations outside their domain.
        for _ in 0..80 {
            let middle = lower + (upper - lower) / 2.0;
            if middle == lower || middle == upper {
                break;
            }
            let sum: f64 = output
                .iter()
                .zip(self.response)
                .map(|(&q, &y)| (q - middle).clamp((y - 1.0) / n, y / n))
                .sum();
            if sum > 0.0 {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        let shift = lower + (upper - lower) / 2.0;
        for (theta, &y) in output.iter_mut().zip(self.response) {
            *theta = (*theta - shift).clamp((y - 1.0) / n, y / n);
        }
        // Repair the last few rounding bits using available box slack.
        for _ in 0..3 {
            let mut correction = -output.iter().sum::<f64>();
            for (theta, &y) in output.iter_mut().zip(self.response) {
                let delta = correction.clamp((y - 1.0) / n - *theta, y / n - *theta);
                let previous = *theta;
                *theta = (*theta + delta).clamp((y - 1.0) / n, y / n);
                correction -= *theta - previous;
                if correction == 0.0 {
                    break;
                }
            }
        }
        self.finalize_dual(output)
    }

    fn finalize_dual(&self, dual: &mut [f64]) -> Result<(), FitError> {
        if dual.len() != self.response.len() || dual.is_empty() {
            return Err(invalid("dual dimensions must match binomial observations"));
        }
        if self.fit_intercept {
            crate::screening::intercept::balance(dual, Some(self.response))?;
        }
        Ok(())
    }

    fn dual_objective(&self, dual: &[f64]) -> Result<f64, FitError> {
        if dual.len() != self.response.len() || dual.is_empty() {
            return Err(invalid("dual dimensions must match binomial observations"));
        }
        let n = dual.len() as f64;
        if self.fit_intercept && dual.iter().sum::<f64>() != 0.0 {
            return Err(invalid(
                "binomial dual point violates the free-intercept equality",
            ));
        }
        let mut value = 0.0;
        for (&theta, &y) in dual.iter().zip(self.response) {
            if !theta.is_finite() || theta < (y - 1.0) / n || theta > y / n {
                return Err(invalid("binomial dual point is outside its entropy domain"));
            }
            let tail = (y - n * theta).min((1.0 - y) + n * theta).clamp(0.0, 0.5);
            value -= negative_entropy(tail) / n;
        }
        Ok(finite(value, "evaluating the binomial dual", 0)?)
    }

    fn strong_concavity(&self) -> f64 {
        4.0 * self.response.len() as f64
    }

    fn fenchel_gap(&self, predictor: &[f64], dual: &[f64]) -> Result<f64, FitError> {
        self.validate(predictor)?;
        self.dual_objective(dual)?;
        let n = predictor.len() as f64;
        let mut gap = 0.0;
        for ((&eta, &y), &theta) in predictor.iter().zip(self.response).zip(dual) {
            // Compute both probabilities from the dual variable. Subtracting
            // from a rounded probability of one would discard a tiny tail.
            let p = (y - n * theta).clamp(0.0, 1.0);
            let q = ((1.0 - y) + n * theta).clamp(0.0, 1.0);
            let tail = (-eta.abs()).exp().ln_1p();
            let cross_entropy = if eta >= 0.0 {
                q * eta + tail
            } else {
                -p * eta + tail
            };
            gap += (cross_entropy + negative_entropy(p.min(q))) / n;
        }
        Ok(finite(gap, "computing the binomial Fenchel gap", 0)?)
    }
}

fn negative_entropy(tail: f64) -> f64 {
    if tail == 0.0 {
        0.0
    } else {
        tail * tail.ln() + (1.0 - tail) * (-tail).ln_1p()
    }
}

impl DiagonalHessianDatafit for Binomial<'_> {
    fn hessian_diagonal(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.validate(predictor)?;
        if output.len() != predictor.len() {
            return Err(invalid("predictor and Hessian lengths must match"));
        }
        for (&eta, curvature) in predictor.iter().zip(output) {
            // Using the small tail avoids losing curvature when the fitted
            // probability rounds to one.
            let tail = (-eta.abs()).exp();
            *curvature = tail / (1.0 + tail).powi(2) / predictor.len() as f64;
        }
        Ok(())
    }
}
