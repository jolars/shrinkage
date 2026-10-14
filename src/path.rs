//! Regularization sequences and reusable path-fitting capabilities.

mod coordinate;
mod datafit;
mod penalty;
mod proximal;

pub use datafit::{NullModelDatafit, PathDatafit};
pub use penalty::{ErasedPathPenalty, PathPenalty};

use crate::FitError;
use crate::error::{finite, invalid};

/// The smallest strength at which the zero-coefficient model is optimal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PenaltyMaximum {
    /// A finite threshold, including zero when the null gradient is zero.
    Finite(f64),
    /// No finite strength makes the zero-coefficient model optimal.
    Unbounded,
}

impl PenaltyMaximum {
    /// Return the finite threshold, or `None` when no finite threshold exists.
    pub fn finite(self) -> Option<f64> {
        match self {
            Self::Finite(value) => Some(value),
            Self::Unbounded => None,
        }
    }
}

/// Options for a decreasing, log-spaced automatic regularization sequence.
///
/// Defaults to 100 points and an endpoint ratio of `0.01` when observations
/// are fewer than features, or `0.0001` otherwise. Every requested point is
/// fitted, without deviance-based early truncation. Options are checked when
/// fitting. A zero starting value produces a sequence of zeros.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct PathOptions {
    nlambda: usize,
    lambda_min_ratio: Option<f64>,
    lambda_start: Option<f64>,
}

impl PathOptions {
    /// Use the default sequence length and dimension-dependent endpoint ratio.
    pub fn new() -> Self {
        Self {
            nlambda: 100,
            lambda_min_ratio: None,
            lambda_start: None,
        }
    }

    /// Set a strictly positive number of points; one returns the starting value.
    pub fn nlambda(mut self, count: usize) -> Self {
        self.nlambda = count;
        self
    }

    /// Set the finite endpoint ratio in `(0, 1]`.
    pub fn lambda_min_ratio(mut self, ratio: f64) -> Self {
        self.lambda_min_ratio = Some(ratio);
        self
    }

    /// Override the starting value with a finite, nonnegative strength.
    /// The result still records the independently calculated true threshold.
    pub fn lambda_start(mut self, value: f64) -> Self {
        self.lambda_start = Some(value);
        self
    }

    pub(crate) fn validate<E>(&self) -> Result<(), FitError<E>> {
        if self.nlambda == 0 {
            return Err(invalid("nlambda must be strictly positive"));
        }
        if self
            .lambda_min_ratio
            .is_some_and(|r| !r.is_finite() || r <= 0.0 || r > 1.0)
        {
            return Err(invalid("lambda_min_ratio must be finite and in (0, 1]"));
        }
        if self.lambda_start.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err(invalid("lambda_start must be finite and nonnegative"));
        }
        Ok(())
    }

    pub(crate) fn sequence<P: PathPenalty>(
        &self,
        penalty: &P,
        gradient: &[f64],
        nobs: usize,
        nfeatures: usize,
    ) -> Result<Sequence<P::AtStrength>, FitError> {
        self.validate()?;
        let maximum = penalty.maximum_penalty(gradient)?;
        let start = match self.lambda_start {
            Some(value) => value,
            None => penalty.path_anchor(gradient)?,
        };
        if !start.is_finite() || start < 0.0 {
            return Err(invalid(
                "penalty path anchor must be finite and nonnegative",
            ));
        }
        let ratio = self
            .lambda_min_ratio
            .unwrap_or(if nobs < nfeatures { 0.01 } else { 0.0001 });
        let mut lambdas = Vec::with_capacity(self.nlambda);
        for point in 0..self.nlambda {
            let value = if point == 0 || start == 0.0 {
                start
            } else if point + 1 == self.nlambda {
                start * ratio
            } else {
                start * (ratio.ln() * point as f64 / (self.nlambda - 1) as f64).exp()
            };
            lambdas.push(finite(value, "generating a regularization sequence", 0)?);
        }
        let penalties = lambdas
            .iter()
            .map(|&lambda| penalty.at_strength(lambda))
            .collect::<Result<_, _>>()?;
        Ok(Sequence {
            penalties,
            lambdas,
            maximum,
            start,
        })
    }
}

impl Default for PathOptions {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) struct Sequence<P> {
    pub penalties: Vec<P>,
    lambdas: Vec<f64>,
    maximum: PenaltyMaximum,
    start: f64,
}

/// Owned fits and their penalty settings in the order they were fitted.
///
/// Parameters and diagnostics belong to each fit. Automatic paths additionally
/// retain scalar strengths and distinguish their starting anchor from a true
/// zero-coefficient threshold. Arbitrary explicit paths have no inferred
/// scalar tuning parameter; inspect their complete penalty settings instead.
#[derive(Clone, Debug)]
#[must_use]
pub struct PathFit<F, P> {
    fits: Vec<F>,
    penalties: Vec<P>,
    lambdas: Option<Vec<f64>>,
    maximum: Option<PenaltyMaximum>,
    start: Option<f64>,
}

impl<F, P> PathFit<F, P> {
    pub(crate) fn explicit(fits: Vec<F>, penalties: Vec<P>) -> Self {
        Self {
            fits,
            penalties,
            lambdas: None,
            maximum: None,
            start: None,
        }
    }

    pub(crate) fn automatic(fits: Vec<F>, sequence: Sequence<P>) -> Self {
        Self {
            fits,
            penalties: sequence.penalties,
            lambdas: Some(sequence.lambdas),
            maximum: Some(sequence.maximum),
            start: Some(sequence.start),
        }
    }

    pub(crate) fn with_lambdas(mut self, lambdas: Vec<f64>) -> Self {
        self.lambdas = Some(lambdas);
        self
    }

    /// Fitted models, including finite results that reached their iteration limit.
    pub fn fits(&self) -> &[F] {
        &self.fits
    }
    /// Complete penalty settings for each fitted point.
    pub fn penalties(&self) -> &[P] {
        &self.penalties
    }
    /// Scalar strengths for generated paths and explicit lasso lambda paths.
    pub fn lambdas(&self) -> Option<&[f64]> {
        self.lambdas.as_deref()
    }
    /// True zero-coefficient threshold, calculated for automatic paths.
    pub fn maximum_penalty(&self) -> Option<PenaltyMaximum> {
        self.maximum
    }
    /// Starting anchor used for an automatic path, including a caller override.
    pub fn lambda_start(&self) -> Option<f64> {
        self.start
    }
    /// Number of fitted points.
    pub fn len(&self) -> usize {
        self.fits.len()
    }
    /// Whether the path contains no points.
    pub fn is_empty(&self) -> bool {
        self.fits.is_empty()
    }
    /// Iterate over corresponding penalty settings and fits.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&P, &F)> {
        self.penalties.iter().zip(&self.fits)
    }
    /// Consume the result and recover its fits and complete penalty settings.
    pub fn into_parts(self) -> (Vec<F>, Vec<P>) {
        (self.fits, self.penalties)
    }
}

/// Fit an explicit penalty sequence using one prepared problem and workspace.
pub trait PathSolver<Problem, P> {
    /// Owned model at each point.
    type Fit;
    /// Validation, component, numerical, or backend failure.
    type Error;
    /// Fit points in caller order, retaining iteration-limit warm starts.
    ///
    /// # Errors
    /// Returns errors without a partial path. No penalty cloning is required.
    fn solve_path(
        &self,
        problem: &Problem,
        penalties: Vec<P>,
    ) -> Result<PathFit<Self::Fit, P>, Self::Error>;
}

/// Generate and fit a family-specific regularization path with one preparation.
pub trait RegularizationPathSolver<Problem> {
    /// Owned model at each point.
    type Fit;
    /// Complete penalty type retained in the result.
    type Penalty;
    /// Validation, component, numerical, or backend failure.
    type Error;
    /// Calculate the true zero-coefficient threshold on the optimization scale.
    ///
    /// # Errors
    /// Reject undefined null models or invalid components and preserve backend errors.
    fn maximum_penalty(&self, problem: &Problem) -> Result<PenaltyMaximum, Self::Error>;
    /// Generate and fit every point, starting from the optimized null intercepts.
    ///
    /// # Errors
    /// Reject invalid sequence options or fitting failures without a partial path.
    fn solve_regularization_path(
        &self,
        problem: &Problem,
        options: &PathOptions,
    ) -> Result<PathFit<Self::Fit, Self::Penalty>, Self::Error>;
}
