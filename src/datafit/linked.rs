//! Family-specific evaluation with statically composed links.

use crate::error::{finite, invalid};
use crate::{
    Binomial, ConvexLink, Datafit, DiagonalHessianDatafit, DifferentiableDatafit, FitError,
    Gaussian, Link, NullModelDatafit, QuadraticModelDatafit, TwiceDifferentiableLink,
};

/// A response family composed with a link for Fisher-scoring IRLS.
///
/// Construct with [`Gaussian::with_link`] or [`Binomial::with_link`]. Explicit
/// IRLS paths require only [`Link`]; automatic paths additionally require
/// [`ConvexLink`]. Exact Hessians require [`TwiceDifferentiableLink`]. The
/// original family constructors retain their specialized solver capabilities.
///
/// Automatic paths require convexity of the particular family-link pairing:
///
/// ```compile_fail
/// use shrinkage::{Gaussian, Logit, NullModelDatafit};
/// fn automatic<D: NullModelDatafit>(_: D) {}
/// automatic(Gaussian::new(&[0.2, 0.8]).with_link(Logit));
/// ```
///
/// A composed link does not acquire proximal solver capabilities implicitly:
///
/// ```compile_fail
/// use shrinkage::{Binomial, Probit, SmoothDatafit};
/// fn proximal<D: SmoothDatafit>(_: D) {}
/// proximal(Binomial::new(&[0.0, 1.0]).with_link(Probit));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct LinkedDatafit<F, L> {
    family: F,
    link: L,
}

impl<F, L> LinkedDatafit<F, L> {
    pub(crate) fn new(family: F, link: L) -> Self {
        Self { family, link }
    }
    /// Borrow the configured link.
    pub fn link_function(&self) -> &L {
        &self.link
    }
    /// Replace the link, preserving the response and intercept setting.
    pub fn with_link<T: Link>(self, link: T) -> LinkedDatafit<F, T> {
        LinkedDatafit::new(self.family, link)
    }
}

impl<F, L: Link> LinkedDatafit<F, L> {
    /// Transform a mean to a predictor using the selected link.
    ///
    /// # Errors
    /// Reject invalid means, link configuration, or arithmetic.
    pub fn link(&self, mu: f64) -> Result<f64, FitError> {
        self.link.validate()?;
        self.link.link(mu)
    }
    /// Transform a predictor to a response mean using the selected link.
    ///
    /// # Errors
    /// Reject invalid predictors, link configuration, or arithmetic.
    pub fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        self.link.validate()?;
        self.link.inverse_link(eta)
    }
}

macro_rules! metadata {
    ($family:ident, $binomial:literal, $reference:expr) => {
        impl<'a, L: Link> LinkedDatafit<$family<'a>, L> {
            /// Enable or disable the free, unpenalized intercept.
            pub fn fit_intercept(mut self, enabled: bool) -> Self {
                self.family = self.family.fit_intercept(enabled);
                self
            }
            /// Borrow the observed responses.
            pub fn response(&self) -> &[f64] {
                self.family.response()
            }
            /// Whether fitting includes an unpenalized intercept.
            pub fn fits_intercept(&self) -> bool {
                self.family.fits_intercept()
            }
            fn validate_response(&self) -> Result<(), FitError> {
                self.link.validate()?;
                if self.response().is_empty() {
                    return Err(invalid("training data must have at least one observation"));
                }
                for (i, &y) in self.response().iter().enumerate() {
                    if !y.is_finite() || ($binomial && !(0.0..=1.0).contains(&y)) {
                        return Err(invalid(format!("invalid response at row {i}")));
                    }
                }
                Ok(())
            }
            fn validate(&self, eta: &[f64]) -> Result<(), FitError> {
                self.validate_response()?;
                if eta.len() != self.response().len() {
                    return Err(invalid("response and predictor lengths must match"));
                }
                if !self.valid_predictor(eta) {
                    return Err(invalid("predictor is outside the family or link domain"));
                }
                Ok(())
            }
            fn initialize(&self) -> Result<f64, FitError> {
                self.validate_response()?;
                if !self.fits_intercept() {
                    return Ok(0.0);
                }
                let mean: f64 = self
                    .response()
                    .iter()
                    .map(|y| y / self.response().len() as f64)
                    .sum();
                for mu in [mean, $reference] {
                    if let Ok(eta) = self.link.link(mu) {
                        if eta.is_finite() && self.valid_predictor(&[eta]) {
                            return Ok(eta);
                        }
                    }
                }
                let eta = self.link.initial_predictor()?;
                if !eta.is_finite() || !self.valid_predictor(&[eta]) {
                    return Err(invalid("link has no valid finite initial intercept"));
                }
                Ok(eta)
            }
        }
    };
}
metadata!(Gaussian, false, 0.0);
metadata!(Binomial, true, 0.5);

fn output_length(eta: &[f64], out: &[f64]) -> Result<(), FitError> {
    if eta.len() != out.len() {
        return Err(invalid("predictor and output lengths must match"));
    }
    Ok(())
}

impl<L: Link> Datafit for LinkedDatafit<Gaussian<'_>, L> {
    type Error = FitError;
    fn value(&self, eta: &[f64]) -> Result<f64, FitError> {
        self.validate(eta)?;
        let mut value = 0.0;
        for (&eta, &y) in eta.iter().zip(self.response()) {
            let residual = finite(
                self.link.inverse_link(eta)? - y,
                "computing a linked Gaussian residual",
                0,
            )?;
            value += (residual / self.n_obs() as f64) * residual * 0.5;
        }
        Ok(finite(value, "computing linked Gaussian loss", 0)?)
    }
}

impl<L: Link> DifferentiableDatafit for LinkedDatafit<Gaussian<'_>, L> {
    fn n_obs(&self) -> usize {
        self.response().len()
    }
    fn fits_intercept(&self) -> bool {
        self.fits_intercept()
    }
    fn gradient(&self, eta: &[f64], out: &mut [f64]) -> Result<(), FitError> {
        self.validate(eta)?;
        output_length(eta, out)?;
        for ((&eta, &y), out) in eta.iter().zip(self.response()).zip(out) {
            *out = finite(
                (self.link.inverse_link(eta)? - y) * self.link.inverse_link_derivative(eta)?
                    / self.n_obs() as f64,
                "computing a linked Gaussian gradient",
                0,
            )?;
        }
        Ok(())
    }
}
impl<L: Link> QuadraticModelDatafit for LinkedDatafit<Gaussian<'_>, L> {
    fn working_curvature(&self, eta: &[f64], out: &mut [f64]) -> Result<(), FitError> {
        self.validate(eta)?;
        output_length(eta, out)?;
        for (&eta, out) in eta.iter().zip(out) {
            let derivative = self.link.inverse_link_derivative(eta)?;
            *out = finite(
                derivative * derivative / self.n_obs() as f64,
                "computing Gaussian Fisher curvature",
                0,
            )?;
        }
        Ok(())
    }
    fn initial_intercept(&self) -> Result<f64, FitError> {
        self.initialize()
    }
    fn valid_predictor(&self, eta: &[f64]) -> bool {
        eta.iter().all(|&eta| {
            self.link.valid_eta(eta) && self.link.inverse_link(eta).is_ok_and(f64::is_finite)
        })
    }
}
impl<L: TwiceDifferentiableLink> DiagonalHessianDatafit for LinkedDatafit<Gaussian<'_>, L> {
    fn hessian_diagonal(&self, eta: &[f64], out: &mut [f64]) -> Result<(), FitError> {
        self.validate(eta)?;
        output_length(eta, out)?;
        for ((&eta, &y), out) in eta.iter().zip(self.response()).zip(out) {
            let derivative = self.link.inverse_link_derivative(eta)?;
            *out = finite(
                (derivative * derivative
                    + (self.link.inverse_link(eta)? - y)
                        * self.link.inverse_link_second_derivative(eta)?)
                    / self.n_obs() as f64,
                "computing a linked Gaussian Hessian",
                0,
            )?;
        }
        Ok(())
    }
}

impl<L: Link> LinkedDatafit<Binomial<'_>, L> {
    fn row(&self, eta: f64, y: f64) -> Result<(f64, f64, f64), FitError> {
        let (log_p, log_q) = self.link.log_probabilities(eta)?;
        let log_d = self.link.log_abs_inverse_link_derivative(eta)?;
        let sign = self.link.inverse_link_derivative(eta)?.signum();
        let loss = -y * log_p - (1.0 - y) * log_q;
        let gradient = sign * ((1.0 - y) * (log_d - log_q).exp() - y * (log_d - log_p).exp());
        let curvature = (2.0 * log_d - log_p - log_q).exp();
        Ok((
            finite(loss, "computing linked binomial loss", 0)?,
            finite(gradient, "computing a linked binomial gradient", 0)?,
            finite(curvature, "computing binomial Fisher curvature", 0)?,
        ))
    }
}
impl<L: Link> Datafit for LinkedDatafit<Binomial<'_>, L> {
    type Error = FitError;
    fn value(&self, eta: &[f64]) -> Result<f64, FitError> {
        self.validate(eta)?;
        let mut value = 0.0;
        for (&eta, &y) in eta.iter().zip(self.response()) {
            value += self.row(eta, y)?.0 / self.n_obs() as f64;
        }
        Ok(finite(value, "summing linked binomial loss", 0)?)
    }
}
impl<L: Link> DifferentiableDatafit for LinkedDatafit<Binomial<'_>, L> {
    fn n_obs(&self) -> usize {
        self.response().len()
    }
    fn fits_intercept(&self) -> bool {
        self.fits_intercept()
    }
    fn gradient(&self, eta: &[f64], out: &mut [f64]) -> Result<(), FitError> {
        self.validate(eta)?;
        output_length(eta, out)?;
        for ((&eta, &y), out) in eta.iter().zip(self.response()).zip(out) {
            *out = self.row(eta, y)?.1 / self.n_obs() as f64;
        }
        Ok(())
    }
}
impl<L: Link> QuadraticModelDatafit for LinkedDatafit<Binomial<'_>, L> {
    fn working_curvature(&self, eta: &[f64], out: &mut [f64]) -> Result<(), FitError> {
        self.validate(eta)?;
        output_length(eta, out)?;
        for (&eta, out) in eta.iter().zip(out) {
            *out = self.row(eta, 0.5)?.2 / self.n_obs() as f64;
        }
        Ok(())
    }
    fn initial_intercept(&self) -> Result<f64, FitError> {
        self.initialize()
    }
    fn valid_predictor(&self, eta: &[f64]) -> bool {
        eta.iter().all(|&eta| {
            self.link.valid_eta(eta)
                && self
                    .link
                    .inverse_link(eta)
                    .is_ok_and(|mu| mu.is_finite() && (0.0..=1.0).contains(&mu))
                && self
                    .link
                    .log_probabilities(eta)
                    .is_ok_and(|(p, q)| p.is_finite() && q.is_finite() && p <= 0.0 && q <= 0.0)
        })
    }
}
impl<L: TwiceDifferentiableLink> DiagonalHessianDatafit for LinkedDatafit<Binomial<'_>, L> {
    fn hessian_diagonal(&self, eta: &[f64], out: &mut [f64]) -> Result<(), FitError> {
        self.validate(eta)?;
        output_length(eta, out)?;
        for ((&eta, &y), out) in eta.iter().zip(self.response()).zip(out) {
            let (p, q) = self.link.log_probabilities(eta)?;
            let d = self.link.log_abs_inverse_link_derivative(eta)?;
            let gradient = self.row(eta, y)?.1;
            if d == f64::NEG_INFINITY {
                // A monotone inverse can have a zero derivative at an interior
                // point. Its exact Hessian does not require dividing by zero.
                let second = self.link.inverse_link_second_derivative(eta)?;
                *out = finite(
                    if second == 0.0 {
                        0.0
                    } else {
                        second * ((1.0 - y) * (-q).exp() - y * (-p).exp()) / self.n_obs() as f64
                    },
                    "computing a linked binomial Hessian",
                    0,
                )?;
                continue;
            }
            *out = finite(
                (y * (2.0 * (d - p)).exp()
                    + (1.0 - y) * (2.0 * (d - q)).exp()
                    + gradient * self.link.inverse_link_derivative_ratio(eta)?)
                    / self.n_obs() as f64,
                "computing a linked binomial Hessian",
                0,
            )?;
        }
        Ok(())
    }
}

impl<'a, L: ConvexLink<Gaussian<'a>>> NullModelDatafit for LinkedDatafit<Gaussian<'a>, L> {
    fn null_intercepts(&self) -> Result<Vec<f64>, FitError> {
        self.validate_response()?;
        let eta = if self.fits_intercept() {
            self.link.link(
                self.response()
                    .iter()
                    .map(|y| y / self.n_obs() as f64)
                    .sum(),
            )?
        } else {
            0.0
        };
        self.value(&vec![eta; self.n_obs()])?;
        Ok(vec![eta])
    }
    fn deviance(&self, eta: &[f64]) -> Result<Option<f64>, FitError> {
        Ok(Some(finite(
            2.0 * self.value(eta)?,
            "computing Gaussian deviance",
            0,
        )?))
    }
}
impl<'a, L: ConvexLink<Binomial<'a>>> NullModelDatafit for LinkedDatafit<Binomial<'a>, L> {
    fn null_intercepts(&self) -> Result<Vec<f64>, FitError> {
        self.validate_response()?;
        if !self.fits_intercept() {
            self.value(&vec![0.0; self.n_obs()])?;
            return Ok(vec![0.0]);
        }
        let successes: f64 = self.response().iter().sum();
        let failures: f64 = self.response().iter().map(|y| 1.0 - y).sum();
        if successes <= 0.0 || failures <= 0.0 {
            return Err(invalid(
                "automatic binomial paths require a finite null intercept; use an explicit path for constant boundary responses",
            ));
        }
        let n = (self.n_obs() as f64).ln();
        let eta = finite(
            self.link
                .link_from_log_probabilities(successes.ln() - n, failures.ln() - n)?,
            "computing a linked binomial null intercept",
            0,
        )?;
        self.value(&vec![eta; self.n_obs()])?;
        Ok(vec![eta])
    }
    fn deviance(&self, eta: &[f64]) -> Result<Option<f64>, FitError> {
        self.validate(eta)?;
        let mut value = 0.0;
        for (&eta, &y) in eta.iter().zip(self.response()) {
            let entropy = if y == 0.0 || y == 1.0 {
                0.0
            } else {
                y * y.ln() + (1.0 - y) * (-y).ln_1p()
            };
            value += (self.row(eta, y)?.0 + entropy).max(0.0) / self.n_obs() as f64;
        }
        Ok(Some(finite(
            2.0 * value,
            "computing linked binomial deviance",
            0,
        )?))
    }
}
