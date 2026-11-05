//! Static link functions and optional second derivatives for GLM composition.

use crate::FitError;
use crate::error::{finite, invalid};

/// A differentiable, monotone link between a mean and a linear predictor.
///
/// Implementations must keep the forward and inverse operations consistent on
/// their domains. Derivatives are with respect to the predictor. Override the
/// log-scale helpers when ordinary probabilities or derivatives lose tail bits.
/// Runtime callbacks belong outside observation loops; Rust links use static
/// dispatch and do not need to implement `Clone` or `Copy`.
pub trait Link {
    /// Transform a mean to a finite predictor.
    ///
    /// # Errors
    /// Reject means outside the link domain or nonfinite arithmetic.
    fn link(&self, mu: f64) -> Result<f64, FitError>;
    /// Transform a finite predictor to a mean.
    ///
    /// # Errors
    /// Reject predictors outside the link domain or nonfinite arithmetic.
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError>;
    /// Evaluate `d mu / d eta`, retaining its sign even after underflow.
    ///
    /// # Errors
    /// Reject invalid predictors or nonfinite arithmetic.
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError>;
    /// Validate parameters independently of any predictor-domain check.
    ///
    /// # Errors
    /// Reject invalid link configuration.
    fn validate(&self) -> Result<(), FitError> {
        Ok(())
    }
    /// Test the predictor domain without evaluating the loss.
    fn valid_eta(&self, eta: f64) -> bool {
        eta.is_finite()
    }
    /// Supply a fallback predictor when family reference means are unavailable.
    ///
    /// # Errors
    /// Reject configurations without a finite initial predictor.
    fn initial_predictor(&self) -> Result<f64, FitError> {
        Ok(0.0)
    }
    /// Return `log(mu)` and `log(1 - mu)` for binomial calculations.
    ///
    /// # Errors
    /// Reject predictors without an interior probability or finite log tails.
    fn log_probabilities(&self, eta: f64) -> Result<(f64, f64), FitError> {
        let mu = self.inverse_link(eta)?;
        probability(mu)?;
        Ok((mu.ln(), (-mu).ln_1p()))
    }
    /// Return `log(abs(d mu / d eta))`; negative infinity represents zero.
    ///
    /// # Errors
    /// Reject invalid predictors or derivatives.
    fn log_abs_inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(finite(
            self.inverse_link_derivative(eta)?,
            "evaluating a link derivative",
            0,
        )?
        .abs()
        .ln())
    }
    /// Transform normalized log probabilities to a predictor.
    ///
    /// Override the default mean conversion to retain rare probability tails.
    ///
    /// # Errors
    /// Reject invalid means or a nonfinite predictor.
    fn link_from_log_probabilities(&self, log_p: f64, _log_q: f64) -> Result<f64, FitError> {
        self.link(log_p.exp())
    }
}

/// A link that also supplies the exact second derivative of its inverse.
pub trait TwiceDifferentiableLink: Link {
    /// Evaluate `d² mu / d eta²`.
    ///
    /// # Errors
    /// Reject invalid predictors or nonfinite arithmetic.
    fn inverse_link_second_derivative(&self, eta: f64) -> Result<f64, FitError>;
    /// Evaluate `mu_second / mu_prime`.
    ///
    /// Override the default quotient when the derivatives underflow separately.
    ///
    /// # Errors
    /// Reject invalid predictors or a nonfinite ratio.
    fn inverse_link_derivative_ratio(&self, eta: f64) -> Result<f64, FitError> {
        Ok(finite(
            self.inverse_link_second_derivative(eta)? / self.inverse_link_derivative(eta)?,
            "evaluating a second derivative ratio",
            0,
        )?)
    }
}

/// Opt into convexity of the loss for a particular family and link.
///
/// Implementors assert convexity on the predictor domain, which must itself be
/// convex. This enables automatic paths for [`crate::LinkedDatafit`]. It does
/// not enable a dual certificate, safe screening, or another solver.
pub trait ConvexLink<F>: Link {}

/// Identity link, `eta = mu`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;
impl Link for Identity {
    fn link(&self, mu: f64) -> Result<f64, FitError> {
        Ok(finite(mu, "evaluating the identity link", 0)?)
    }
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        self.link(eta)
    }
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        self.link(eta)?;
        Ok(1.0)
    }
}
impl TwiceDifferentiableLink for Identity {
    fn inverse_link_second_derivative(&self, eta: f64) -> Result<f64, FitError> {
        self.link(eta)?;
        Ok(0.0)
    }
    fn inverse_link_derivative_ratio(&self, eta: f64) -> Result<f64, FitError> {
        self.inverse_link_second_derivative(eta)
    }
}
impl ConvexLink<crate::Gaussian<'_>> for Identity {}

/// Logit link, `eta = log(mu / (1 - mu))`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Logit;
impl Link for Logit {
    fn link(&self, mu: f64) -> Result<f64, FitError> {
        probability(mu)?;
        Ok(mu.ln() - (-mu).ln_1p())
    }
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        finite(eta, "evaluating the inverse logit", 0)?;
        let tail = (-eta.abs()).exp();
        Ok(if eta >= 0.0 {
            1.0 / (1.0 + tail)
        } else {
            tail / (1.0 + tail)
        })
    }
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        finite(eta, "evaluating the inverse logit derivative", 0)?;
        let tail = (-eta.abs()).exp();
        Ok(tail / (1.0 + tail).powi(2))
    }
    fn log_probabilities(&self, eta: f64) -> Result<(f64, f64), FitError> {
        finite(eta, "evaluating logit log probabilities", 0)?;
        let tail = (-eta.abs()).exp().ln_1p();
        Ok(if eta >= 0.0 {
            (-tail, -eta - tail)
        } else {
            (eta - tail, -tail)
        })
    }
    fn log_abs_inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        finite(eta, "evaluating the log inverse logit derivative", 0)?;
        Ok(-eta.abs() - 2.0 * (-eta.abs()).exp().ln_1p())
    }
    fn link_from_log_probabilities(&self, log_p: f64, log_q: f64) -> Result<f64, FitError> {
        Ok(finite(log_p - log_q, "transforming log probabilities", 0)?)
    }
}
impl TwiceDifferentiableLink for Logit {
    fn inverse_link_second_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(self.inverse_link_derivative(eta)? * self.inverse_link_derivative_ratio(eta)?)
    }
    fn inverse_link_derivative_ratio(&self, eta: f64) -> Result<f64, FitError> {
        Ok(1.0 - 2.0 * self.inverse_link(eta)?)
    }
}
impl ConvexLink<crate::Binomial<'_>> for Logit {}

/// Probit link, the standard normal quantile function.
#[derive(Clone, Copy, Debug, Default)]
pub struct Probit;
impl Link for Probit {
    fn link(&self, mu: f64) -> Result<f64, FitError> {
        probability(mu)?;
        self.link_from_log_probabilities(mu.ln(), (-mu).ln_1p())
    }
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        finite(eta, "evaluating the inverse probit", 0)?;
        Ok(0.5 * libm::erfc(-eta / std::f64::consts::SQRT_2))
    }
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(self.log_abs_inverse_link_derivative(eta)?.exp())
    }
    fn log_probabilities(&self, eta: f64) -> Result<(f64, f64), FitError> {
        finite(eta, "evaluating probit log probabilities", 0)?;
        Ok((
            finite(log_normal_cdf(eta), "evaluating a normal log tail", 0)?,
            finite(log_normal_cdf(-eta), "evaluating a normal log tail", 0)?,
        ))
    }
    fn log_abs_inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        finite(eta, "evaluating the log inverse probit derivative", 0)?;
        Ok(finite(
            -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln(),
            "evaluating the normal log density",
            0,
        )?)
    }
    fn link_from_log_probabilities(&self, log_p: f64, log_q: f64) -> Result<f64, FitError> {
        if !log_p.is_finite()
            || !log_q.is_finite()
            || log_p > 0.0
            || log_q > 0.0
            || (log_p == 0.0 && log_q == 0.0)
        {
            return Err(invalid(
                "probit requires finite log probabilities of an interior mean",
            ));
        }
        if log_p == log_q {
            return Ok(0.0);
        }
        let target = log_p.min(log_q);
        let mut lower = -(-2.0 * target).sqrt() - 1.0;
        let mut upper = 0.0;
        // Quantiles are used for conversions and initialization, outside the
        // solver's observation sweeps. Log-tail bisection also handles subnormals.
        for _ in 0..128 {
            let middle = lower + (upper - lower) / 2.0;
            if middle == lower || middle == upper {
                break;
            }
            if log_normal_cdf(middle) < target {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        let eta = lower + (upper - lower) / 2.0;
        Ok(finite(
            if log_p <= log_q { eta } else { -eta },
            "evaluating the probit link",
            0,
        )?)
    }
}
impl TwiceDifferentiableLink for Probit {
    fn inverse_link_second_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(-eta * self.inverse_link_derivative(eta)?)
    }
    fn inverse_link_derivative_ratio(&self, eta: f64) -> Result<f64, FitError> {
        Ok(-finite(
            eta,
            "evaluating the probit second derivative ratio",
            0,
        )?)
    }
}
impl ConvexLink<crate::Binomial<'_>> for Probit {}

fn probability(mu: f64) -> Result<(), FitError> {
    if !mu.is_finite() || mu <= 0.0 || mu >= 1.0 {
        return Err(invalid(
            "link mean must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn log_normal_cdf(eta: f64) -> f64 {
    if eta > 0.0 {
        return (-log_normal_cdf(-eta).exp()).ln_1p();
    }
    if eta >= -10.0 {
        return (0.5 * libm::erfc(-eta / std::f64::consts::SQRT_2)).ln();
    }
    // Mills' asymptotic series avoids underflow in erfc and the normal density.
    // Stop before divergent terms or after reaching floating-point precision.
    let inverse_square = (1.0 / eta).powi(2);
    let mut sum = 1.0;
    let mut term: f64 = 1.0;
    for k in 1..128 {
        let next = -term * (2 * k - 1) as f64 * inverse_square;
        if next.abs() >= term.abs() {
            break;
        }
        let updated = sum + next;
        if updated == sum {
            break;
        }
        sum = updated;
        term = next;
    }
    -0.5 * eta * eta - (-eta).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() + sum.ln()
}
