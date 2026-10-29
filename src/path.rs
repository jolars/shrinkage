//! Regularization sequences and reusable path-fitting capabilities.

mod coordinate;
mod datafit;
mod irls;
mod penalty;
mod proximal;
pub(crate) use proximal::null_state;

pub use datafit::{NullModelDatafit, PathDatafit};
pub use penalty::{ErasedPathPenalty, PathPenalty};

use crate::error::{finite, invalid};
use crate::{FitError, Termination};

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
/// are fewer than features, or `0.0001` otherwise. Automatic paths stop early
/// on deviance by default, after at least five points. Disable early stopping
/// to fit the entire sequence. Losses without deviance support fit every point.
/// Options are checked when fitting. A zero start generates zero strengths.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct PathOptions {
    nlambda: usize,
    lambda_min_ratio: Option<f64>,
    lambda_start: Option<f64>,
    early_stopping: bool,
    min_points: usize,
    deviance_change_tolerance: f64,
    deviance_ratio_tolerance: f64,
}

impl PathOptions {
    /// Use the default sequence length and dimension-dependent endpoint ratio.
    pub fn new() -> Self {
        Self {
            nlambda: 100,
            lambda_min_ratio: None,
            lambda_start: None,
            early_stopping: true,
            min_points: 5,
            deviance_change_tolerance: 1e-5,
            deviance_ratio_tolerance: 0.999,
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

    /// Enable deviance-based early stopping for automatic paths (the default).
    pub fn early_stopping(mut self, enabled: bool) -> Self {
        self.early_stopping = enabled;
        self
    }

    /// Require this many fitted points before early stopping; defaults to five.
    /// Shorter requested sequences are fitted completely.
    pub fn min_points(mut self, count: usize) -> Self {
        self.min_points = count;
        self
    }

    /// Stop below this fractional decrease in consecutive deviances (`1e-5`).
    /// Zero disables this criterion. Only converged, nonincreasing fits qualify.
    pub fn deviance_change_tolerance(mut self, tolerance: f64) -> Self {
        self.deviance_change_tolerance = tolerance;
        self
    }

    /// Stop at this explained fraction of null deviance; defaults to `0.999`.
    pub fn deviance_ratio_tolerance(mut self, tolerance: f64) -> Self {
        self.deviance_ratio_tolerance = tolerance;
        self
    }

    pub(crate) fn validate<E>(&self) -> Result<(), FitError<E>> {
        if self.nlambda == 0 {
            return Err(invalid("nlambda must be strictly positive"));
        }
        if self.min_points == 0 {
            return Err(invalid("min_points must be strictly positive"));
        }
        if !self.deviance_change_tolerance.is_finite() || self.deviance_change_tolerance < 0.0 {
            return Err(invalid(
                "deviance_change_tolerance must be finite and nonnegative",
            ));
        }
        if !self.deviance_ratio_tolerance.is_finite()
            || self.deviance_ratio_tolerance <= 0.0
            || self.deviance_ratio_tolerance > 1.0
        {
            return Err(invalid(
                "deviance_ratio_tolerance must be finite and in (0, 1]",
            ));
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
            diagnostics: PathDiagnostics::default(),
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
    diagnostics: PathDiagnostics,
}

/// Why path fitting stopped, independently of each model's convergence status.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PathTermination {
    /// Every requested tuning point was fitted.
    #[default]
    Completed,
    /// Consecutive converged fits improved deviance by less than the tolerance.
    DevianceChange,
    /// A converged fit explained the requested fraction of null deviance.
    DevianceRatio,
}

#[derive(Clone, Debug, Default)]
struct PathDiagnostics {
    termination: PathTermination,
    null_deviance: Option<f64>,
    deviances: Option<Vec<f64>>,
}

pub(crate) fn null_deviance<D: NullModelDatafit + ?Sized, E>(
    datafit: &D,
    nobs: usize,
) -> Result<Option<f64>, FitError<E>> {
    let intercepts = datafit
        .null_intercepts()
        .map_err(|e| FitError::from_component(e, 0))?;
    let predictor = intercepts
        .iter()
        .flat_map(|&intercept| std::iter::repeat_n(intercept, nobs))
        .collect::<Vec<_>>();
    datafit
        .deviance(&predictor)
        .map_err(|e| FitError::from_component(e, 0))
}

impl<P> Sequence<P> {
    pub(crate) fn fit<S, E>(
        &mut self,
        options: &PathOptions,
        null: Option<f64>,
        mut fit: impl FnMut(&P) -> Result<(S, Option<f64>, Termination), FitError<E>>,
    ) -> Result<Vec<S>, FitError<E>> {
        if null.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(invalid("null deviance must be finite and nonnegative"));
        }
        self.diagnostics.null_deviance = null;
        self.diagnostics.deviances = null.map(|_| Vec::with_capacity(self.penalties.len()));
        let mut previous = None;
        let mut solutions = Vec::with_capacity(self.penalties.len());
        for penalty in &self.penalties {
            let (solution, deviance, termination) = fit(penalty)?;
            solutions.push(solution);
            if deviance.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(invalid("path deviance must be finite and nonnegative"));
            }
            let (Some(null), Some(deviances)) = (null, &mut self.diagnostics.deviances) else {
                continue;
            };
            let deviance = deviance.ok_or_else(|| {
                invalid("datafit deviance support must be consistent along a path")
            })?;
            deviances.push(deviance);
            let converged = termination == Termination::Converged;
            if options.early_stopping
                && null > 0.0
                && converged
                && solutions.len() >= options.min_points
                && solutions.len() < self.penalties.len()
            {
                if 1.0 - deviance / null >= options.deviance_ratio_tolerance {
                    self.diagnostics.termination = PathTermination::DevianceRatio;
                    break;
                }
                if let Some((last, true)) = previous {
                    if last > 0.0
                        && deviance <= last
                        && (last - deviance) / last < options.deviance_change_tolerance
                    {
                        self.diagnostics.termination = PathTermination::DevianceChange;
                        break;
                    }
                }
            }
            previous = Some((deviance, converged));
        }
        self.penalties.truncate(solutions.len());
        self.lambdas.truncate(solutions.len());
        Ok(solutions)
    }
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
    diagnostics: PathDiagnostics,
}

impl<F, P> PathFit<F, P> {
    pub(crate) fn explicit(fits: Vec<F>, penalties: Vec<P>) -> Self {
        Self {
            fits,
            penalties,
            lambdas: None,
            maximum: None,
            start: None,
            diagnostics: PathDiagnostics::default(),
        }
    }

    pub(crate) fn automatic(fits: Vec<F>, sequence: Sequence<P>) -> Self {
        Self {
            fits,
            penalties: sequence.penalties,
            lambdas: Some(sequence.lambdas),
            maximum: Some(sequence.maximum),
            start: Some(sequence.start),
            diagnostics: sequence.diagnostics,
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
    /// Path completion or early stopping, independent of per-model convergence.
    pub fn termination(&self) -> PathTermination {
        self.diagnostics.termination
    }
    /// Averaged deviance of the automatic path's optimized zero-coefficient model.
    /// Unavailable for explicit paths and losses without deviance support.
    pub fn null_deviance(&self) -> Option<f64> {
        self.diagnostics.null_deviance
    }
    /// Averaged original-loss deviances, aligned with the fitted automatic points.
    /// Unavailable for explicit paths and losses without deviance support.
    pub fn deviances(&self) -> Option<&[f64]> {
        self.diagnostics.deviances.as_deref()
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
    /// Generate and fit a path, starting from the optimized null intercepts.
    /// Automatic paths may stop early according to [`PathOptions`].
    ///
    /// # Errors
    /// Reject invalid sequence options or fitting failures without a partial path.
    fn solve_regularization_path(
        &self,
        problem: &Problem,
        options: &PathOptions,
    ) -> Result<PathFit<Self::Fit, Self::Penalty>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::L1;

    #[test]
    fn unconverged_points_never_trigger_deviance_stopping() {
        let options = PathOptions::new()
            .nlambda(8)
            .min_points(2)
            .deviance_change_tolerance(0.1)
            .deviance_ratio_tolerance(1.0);
        let mut sequence = options.sequence(&L1::new(1.0), &[1.0], 2, 1).unwrap();
        let mut point = 0;
        let solutions = sequence
            .fit::<_, std::convert::Infallible>(&options, Some(1.0), |_| {
                point += 1;
                let termination = if point == 2 {
                    Termination::IterationLimit
                } else {
                    Termination::Converged
                };
                Ok(((), Some(if point <= 2 { 0.6 } else { 0.59 }), termination))
            })
            .unwrap();
        assert_eq!(solutions.len(), 4);
        assert_eq!(sequence.penalties.len(), 4);
        assert_eq!(
            sequence.diagnostics.termination,
            PathTermination::DevianceChange
        );
        let mut sequence = options.sequence(&L1::new(1.0), &[1.0], 2, 1).unwrap();
        let solutions = sequence
            .fit::<_, std::convert::Infallible>(&options, Some(1.0), |_| {
                Ok(((), Some(0.0), Termination::IterationLimit))
            })
            .unwrap();
        assert_eq!(solutions.len(), 8);
        assert_eq!(sequence.diagnostics.termination, PathTermination::Completed);
    }

    #[test]
    fn invalid_deviances_and_missing_values_are_errors() {
        let options = PathOptions::new().nlambda(2);
        for deviance in [Some(f64::NAN), Some(f64::INFINITY), Some(-1.0), None] {
            let mut sequence = options.sequence(&L1::new(1.0), &[1.0], 2, 1).unwrap();
            assert!(
                sequence
                    .fit::<_, std::convert::Infallible>(&options, Some(1.0), |_| {
                        Ok(((), deviance, Termination::Converged))
                    })
                    .is_err()
            );
        }
    }
}
