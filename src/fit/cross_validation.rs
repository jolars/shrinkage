//! Training-fold paths, held-out scores, selection, and full-data refits.

use crate::{FoldAssignments, PathFit, SelectionRule, ValidationMetric};

/// One training-fold path and its held-out losses in candidate order.
#[derive(Clone, Debug)]
pub struct FoldPath<F, P> {
    pub(crate) validation_rows: Vec<usize>,
    pub(crate) path: PathFit<F, P>,
    pub(crate) scores: Vec<f64>,
}

impl<F, P> FoldPath<F, P> {
    /// Original observation indices held out for this fold, in increasing order.
    pub fn validation_rows(&self) -> &[usize] {
        &self.validation_rows
    }
    /// Fits, training preprocessing, complete penalties, and path diagnostics.
    pub fn path(&self) -> &PathFit<F, P> {
        &self.path
    }
    /// Losses averaged within this fold, excluding all penalty terms.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }
}

/// Fold paths, aggregated uncertainty, selected candidate, and full-data refits.
///
/// Fold scores are averaged with equal fold weights, even for unequal sizes.
/// For `K` folds, `SE = sqrt(sum((score - mean)^2) / (K * (K - 1)))`.
/// These conventional fold standard errors do not account for dependence
/// between overlapping training sets. All candidate fits must converge.
#[derive(Clone, Debug)]
#[must_use]
pub struct CrossValidationFit<F, P> {
    pub(crate) assignments: FoldAssignments,
    pub(crate) folds: Vec<FoldPath<F, P>>,
    pub(crate) refit: PathFit<F, P>,
    pub(crate) mean_scores: Vec<f64>,
    pub(crate) standard_errors: Vec<f64>,
    pub(crate) metric: ValidationMetric,
    pub(crate) selection_rule: SelectionRule,
    pub(crate) minimum: usize,
    pub(crate) one_standard_error: usize,
    pub(crate) selected: usize,
    pub(crate) relative_strengths: Option<Vec<f64>>,
}

impl<F, P> CrossValidationFit<F, P> {
    /// Held-out assignments in original observation order.
    pub fn assignments(&self) -> &FoldAssignments {
        &self.assignments
    }
    /// Training-fold paths in zero-based fold order.
    pub fn folds(&self) -> &[FoldPath<F, P>] {
        &self.folds
    }
    /// Fresh path fitted on all observations after validation and selection.
    pub fn refit_path(&self) -> &PathFit<F, P> {
        &self.refit
    }
    /// Selected fit from the full-data path, with original-scale parameters.
    pub fn fit(&self) -> &F {
        &self.refit.fits()[self.selected]
    }
    /// Selected complete full-data penalty, including every strength and weight.
    pub fn penalty(&self) -> &P {
        &self.refit.penalties()[self.selected]
    }
    /// Selected full-data scalar strength, when the path supplies one.
    pub fn lambda(&self) -> Option<f64> {
        self.refit.lambdas().map(|s| s[self.selected])
    }
    /// Shared automatic fractions of each fold's own starting anchor.
    /// Explicit paths have no inferred scalar representation.
    pub fn relative_strengths(&self) -> Option<&[f64]> {
        self.relative_strengths.as_deref()
    }
    /// Equal-weight means of held-out fold losses, aligned with candidates.
    pub fn mean_scores(&self) -> &[f64] {
        &self.mean_scores
    }
    /// Sample standard deviations of fold losses divided by `sqrt(K)`.
    pub fn standard_errors(&self) -> &[f64] {
        &self.standard_errors
    }
    /// Family-compatible metric used for validation.
    pub fn metric(&self) -> ValidationMetric {
        self.metric
    }
    /// Rule used to select the full-data fit.
    pub fn selection_rule(&self) -> SelectionRule {
        self.selection_rule
    }
    /// First candidate with minimum mean loss.
    pub fn minimum_index(&self) -> usize {
        self.minimum
    }
    /// First candidate within one standard error of the minimum-loss point.
    pub fn one_standard_error_index(&self) -> usize {
        self.one_standard_error
    }
    /// Candidate chosen by the configured rule.
    pub fn selected_index(&self) -> usize {
        self.selected
    }
}
