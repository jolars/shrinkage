//! Training-only regularization paths, held-out scoring, and model selection.
//!
//! Each fold fits preprocessing and an entire warm-start path on its training
//! rows. Explicit candidates retain their complete penalties. Automatic paths
//! align by fractions of independently fitted training-fold anchors and disable
//! early stopping. Selection uses equal-weight fold means and their sample
//! standard errors. Only then is a fresh path fitted on all observations.
//!
//! Explicit candidates must be ordered from most to least preferred for
//! simplicity. Ties choose the first candidate. The one-standard-error rule
//! chooses the first candidate within one standard error of the minimum.
//! This ordering is caller-defined for independent strengths or weight profiles;
//! no scalar ordering is inferred from a multi-parameter penalty.

mod error;
mod folds;
mod metric;
mod rows;

pub use crate::fit::{CrossValidationFit, FoldPath};
pub use error::CrossValidationError;
pub use folds::FoldAssignments;
pub use metric::{CrossValidationDatafit, ValidationMetric};
pub use rows::RowSubset;

use lazymatrix::RawColumns;

use crate::error::{finite, invalid};
use crate::{
    FitError, PathFit, PathOptions, PathPenalty, PathSolver, Problem, RegularizationPathSolver,
    Termination,
};

/// Rule used to select a candidate after all validation paths converge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectionRule {
    /// First candidate attaining the lowest mean held-out loss.
    #[default]
    MinimumError,
    /// First candidate within one standard error of the minimum-loss candidate.
    OneStandardError,
}

/// Validated folds, held-out metric, and selection policy.
#[derive(Clone, Debug)]
#[must_use]
pub struct CrossValidation {
    folds: FoldAssignments,
    metric: Option<ValidationMetric>,
    selection_rule: SelectionRule,
}

impl CrossValidation {
    /// Use supplied folds, the family's default metric, and minimum-error selection.
    pub fn new(folds: FoldAssignments) -> Self {
        Self {
            folds,
            metric: None,
            selection_rule: SelectionRule::MinimumError,
        }
    }
    /// Override the family's default held-out metric, checked when fitting.
    pub fn metric(mut self, metric: ValidationMetric) -> Self {
        self.metric = Some(metric);
        self
    }
    /// Choose minimum error or the one-standard-error preference rule.
    pub fn selection_rule(mut self, rule: SelectionRule) -> Self {
        self.selection_rule = rule;
        self
    }
    /// Borrow the reproducible fold assignments.
    pub fn folds(&self) -> &FoldAssignments {
        &self.folds
    }
}

/// Prediction and convergence capabilities of a univariate fitted model.
pub trait ValidationFit {
    /// Predict original-scale linear predictors from raw held-out columns.
    ///
    /// # Errors
    /// Reject invalid dimensions, entries, or prediction arithmetic.
    fn predict<M: RawColumns<f64> + ?Sized>(&self, design: &M) -> Result<Vec<f64>, FitError>;
    /// Report convergence under the solver's selected stopping criterion.
    fn termination(&self) -> Termination;
}

macro_rules! validation_fit {
    ($($fit:ty),* $(,)?) => {$ (
        impl ValidationFit for $fit {
            fn predict<M: RawColumns<f64> + ?Sized>(&self, design: &M) -> Result<Vec<f64>, FitError> {
                self.predict(design)
            }
            fn termination(&self) -> Termination { self.termination() }
        }
    )*};
}
validation_fit!(
    crate::LassoFit,
    crate::IrlsFit,
    crate::ProximalFit,
    crate::HybridFit
);

impl<M, D, P> Problem<&M, D, P>
where
    M: RowSubset + ?Sized,
    D: CrossValidationDatafit,
{
    /// Cross-validate an explicit sequence, reusing a path workspace per fold.
    ///
    /// Candidates are cloned unchanged into each fold. Order them from most to
    /// least preferred for simplicity to define ties and one-standard-error
    /// selection, including penalties with independent strengths or profiles.
    /// A full-data path is fitted after selection and retained in the result.
    ///
    /// # Errors
    /// Reject invalid folds, metrics, supplied centers or scales, degenerate
    /// binomial training folds, empty paths, and any failed or nonconverged fit.
    pub fn cross_validate_path_with<S, Q, F>(
        &self,
        solver: &S,
        penalties: Vec<Q>,
        cv: &CrossValidation,
    ) -> Result<CrossValidationFit<F, Q>, CrossValidationError<M::Error>>
    where
        Q: Clone,
        F: ValidationFit,
        S: PathSolver<Self, Q, Fit = F, Error = FitError<M::Error>>,
        for<'a> S: PathSolver<
                Problem<&'a M::Subset, D::WithResponse<'a>, ()>,
                Q,
                Fit = F,
                Error = FitError<M::Error>,
            >,
    {
        let metric = self.validate_cross_validation(cv)?;
        if penalties.is_empty() {
            return Err(invalid("cross-validation requires at least one candidate").into());
        }
        let mut folds = Vec::with_capacity(cv.folds.nfolds());
        for fold in 0..cv.folds.nfolds() {
            let result = (|| {
                let (training_rows, validation_rows) = cv.folds.rows(fold);
                let training = self.design().select_rows(&training_rows)?;
                let response = training_rows
                    .iter()
                    .map(|&i| self.datafit().response()[i])
                    .collect::<Vec<_>>();
                let datafit = self.training_datafit(&response)?;
                let problem = Problem::new(&training, datafit, ()).normalize(self.normalization());
                let path = solver.solve_path(&problem, penalties.clone())?;
                self.score_fold(path, validation_rows, metric, penalties.len())
            })()
            .map_err(|source| CrossValidationError::Fold { fold, source })?;
            check_convergence(&result.path, Some(fold))?;
            folds.push(result);
        }
        let summary = summarize(&folds, cv.selection_rule)?;
        let refit = solver
            .solve_path(self, penalties)
            .map_err(|source| CrossValidationError::Refit { source })?;
        finish(cv, metric, folds, refit, summary, None)
    }

    /// Cross-validate automatic paths aligned by relative training-fold strength.
    ///
    /// Each fold computes normalization, null gradients, and its anchor only on
    /// its training rows. Candidates share log-spaced fractions of that anchor.
    /// The dimension-dependent endpoint ratio is resolved once using the
    /// smallest training fold. Early stopping is disabled to score every point.
    /// `lambda_start`, when supplied, is a fixed external anchor for every fold.
    /// The full-data path uses its own anchor after validation selects a fraction.
    ///
    /// # Errors
    /// Return fold-specific failures, undefined anchors, invalid settings, or
    /// any nonconverged path point. Supplied preprocessing vectors are rejected.
    pub fn cross_validate_regularization_path_with<S, F>(
        &self,
        solver: &S,
        options: &PathOptions,
        cv: &CrossValidation,
    ) -> Result<CrossValidationFit<F, P::AtStrength>, CrossValidationError<M::Error>>
    where
        P: PathPenalty + Clone,
        F: ValidationFit,
        S: RegularizationPathSolver<
                Self,
                Fit = F,
                Penalty = P::AtStrength,
                Error = FitError<M::Error>,
            >,
        for<'a> S: RegularizationPathSolver<
                Problem<&'a M::Subset, D::WithResponse<'a>, P>,
                Fit = F,
                Penalty = P::AtStrength,
                Error = FitError<M::Error>,
            >,
    {
        let metric = self.validate_cross_validation(cv)?;
        options.validate()?;
        let min_training = self.design().nrows() - cv.folds.fold_sizes().iter().max().unwrap();
        let (options, fractions) = options.cross_validation(min_training, self.design().ncols());
        let mut folds = Vec::with_capacity(cv.folds.nfolds());
        for fold in 0..cv.folds.nfolds() {
            let result = (|| {
                let (training_rows, validation_rows) = cv.folds.rows(fold);
                let training = self.design().select_rows(&training_rows)?;
                let response = training_rows
                    .iter()
                    .map(|&i| self.datafit().response()[i])
                    .collect::<Vec<_>>();
                let datafit = self.training_datafit(&response)?;
                let problem = Problem::new(&training, datafit, self.penalty().clone())
                    .normalize(self.normalization());
                let path = solver.solve_regularization_path(&problem, &options)?;
                self.score_fold(path, validation_rows, metric, fractions.len())
            })()
            .map_err(|source| CrossValidationError::Fold { fold, source })?;
            check_convergence(&result.path, Some(fold))?;
            folds.push(result);
        }
        let summary = summarize(&folds, cv.selection_rule)?;
        let refit = solver
            .solve_regularization_path(self, &options)
            .map_err(|source| CrossValidationError::Refit { source })?;
        finish(cv, metric, folds, refit, summary, Some(fractions))
    }

    fn validate_cross_validation(
        &self,
        cv: &CrossValidation,
    ) -> Result<ValidationMetric, FitError<M::Error>> {
        if self.design().nrows() != self.datafit().response().len()
            || self.design().nrows() != cv.folds.assignments().len()
        {
            return Err(invalid(
                "design, response, and fold assignment lengths must match",
            ));
        }
        if self.centers().is_some() || self.scales().is_some() {
            return Err(invalid(
                "cross-validation must estimate preprocessing within each training fold; remove supplied centers and scales",
            ));
        }
        self.datafit()
            .validate_training()
            .map_err(|e| FitError::from_component(e, 0))?;
        crate::problem::validate_matrix(*self.design())?;
        let metric = cv
            .metric
            .unwrap_or_else(|| self.datafit().validation_metric());
        if !self.datafit().supports_metric(metric) {
            return Err(invalid(
                "validation metric does not support the problem's response family",
            ));
        }
        Ok(metric)
    }

    fn training_datafit<'a>(
        &self,
        response: &'a [f64],
    ) -> Result<D::WithResponse<'a>, FitError<M::Error>> {
        let datafit = self.datafit().with_response(response);
        datafit
            .validate_training()
            .map_err(|e| FitError::from_component(e, 0))?;
        Ok(datafit)
    }

    fn score_fold<F: ValidationFit, Q>(
        &self,
        path: PathFit<F, Q>,
        validation_rows: Vec<usize>,
        metric: ValidationMetric,
        count: usize,
    ) -> Result<FoldPath<F, Q>, FitError<M::Error>> {
        if path.len() != count {
            return Err(invalid(
                "every cross-validation path must fit the complete candidate sequence",
            ));
        }
        let validation = self.design().select_rows(&validation_rows)?;
        let response = validation_rows
            .iter()
            .map(|&i| self.datafit().response()[i])
            .collect::<Vec<_>>();
        let scores = path
            .fits()
            .iter()
            .map(|fit| {
                let predictor = fit
                    .predict(&validation)
                    .map_err(|e| FitError::from_component(e, 0))?;
                metric
                    .score(&response, &predictor)
                    .map_err(|e| FitError::from_component(e, 0))
            })
            .collect::<Result<_, _>>()?;
        Ok(FoldPath {
            validation_rows,
            path,
            scores,
        })
    }
}

struct Summary {
    means: Vec<f64>,
    errors: Vec<f64>,
    minimum: usize,
    one_standard_error: usize,
    selected: usize,
}

fn summarize<F, P, E>(
    folds: &[FoldPath<F, P>],
    rule: SelectionRule,
) -> Result<Summary, FitError<E>> {
    let k = folds.len() as f64;
    let mut means = Vec::with_capacity(folds[0].scores.len());
    let mut errors = Vec::with_capacity(folds[0].scores.len());
    for candidate in 0..folds[0].scores.len() {
        let mean = finite(
            folds.iter().map(|fold| fold.scores[candidate] / k).sum(),
            "averaging validation scores",
            0,
        )?;
        let divisor = k.sqrt() * (k - 1.0).sqrt();
        let se = folds.iter().fold(0.0_f64, |norm, fold| {
            norm.hypot((fold.scores[candidate] - mean) / divisor)
        });
        means.push(mean);
        errors.push(finite(se, "computing validation standard errors", 0)?);
    }
    let minimum = (0..means.len()).fold(0, |best, j| if means[j] < means[best] { j } else { best });
    let one_standard_error = (0..means.len())
        .find(|&j| means[j] <= means[minimum] || means[j] - means[minimum] <= errors[minimum])
        .unwrap();
    let selected = match rule {
        SelectionRule::MinimumError => minimum,
        SelectionRule::OneStandardError => one_standard_error,
    };
    Ok(Summary {
        means,
        errors,
        minimum,
        one_standard_error,
        selected,
    })
}

fn check_convergence<F: ValidationFit, P, E>(
    path: &PathFit<F, P>,
    fold: Option<usize>,
) -> Result<(), CrossValidationError<E>> {
    for (candidate, fit) in path.fits().iter().enumerate() {
        if fit.termination() != Termination::Converged {
            return Err(CrossValidationError::NonConverged { fold, candidate });
        }
    }
    Ok(())
}

fn finish<F: ValidationFit, P, E>(
    cv: &CrossValidation,
    metric: ValidationMetric,
    folds: Vec<FoldPath<F, P>>,
    refit: PathFit<F, P>,
    summary: Summary,
    fractions: Option<Vec<f64>>,
) -> Result<CrossValidationFit<F, P>, CrossValidationError<E>> {
    if refit.len() != summary.means.len() {
        return Err(CrossValidationError::Refit {
            source: invalid("full-data refit must retain every validation candidate"),
        });
    }
    check_convergence(&refit, None)?;
    Ok(CrossValidationFit {
        assignments: cv.folds.clone(),
        folds,
        refit,
        mean_scores: summary.means,
        standard_errors: summary.errors,
        metric,
        selection_rule: cv.selection_rule,
        minimum: summary.minimum,
        one_standard_error: summary.one_standard_error,
        selected: summary.selected,
        relative_strengths: fractions,
    })
}
