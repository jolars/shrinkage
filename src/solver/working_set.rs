//! Heuristic restrictions, kept separate from certified zero screening.

use crate::FitError;
use crate::error::invalid;

/// Feature selection for restricted numerical updates.
///
/// Every policy retains current nonzeros. Excluded zeros can reenter after a
/// complete optimality check; only the original problem certifies convergence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorkingSetPolicy {
    /// Adaptive sets for single fits, and ever-active sets for paths.
    #[default]
    Auto,
    /// Update every feature allowed by independently configured screening.
    Disabled,
    /// Retain features activated at accepted iterates, including earlier path points.
    EverActive,
    /// Rank features by optimality violation, always retaining current nonzeros.
    Adaptive {
        /// Positive initial feature budget, enlarged to retain the starting support.
        initial_size: usize,
        /// Rebuild around twice the support instead of retaining and doubling the set.
        prune: bool,
    },
}

impl WorkingSetPolicy {
    pub(crate) fn enabled(enabled: bool) -> Self {
        if enabled { Self::Auto } else { Self::Disabled }
    }

    pub(crate) fn resolve(self, path: bool) -> Self {
        match self {
            Self::Auto if path => Self::EverActive,
            Self::Auto => Self::Adaptive {
                initial_size: 10,
                prune: true,
            },
            other => other,
        }
    }

    pub(crate) fn validate<E>(self) -> Result<(), FitError<E>> {
        if matches!(
            self,
            Self::Adaptive {
                initial_size: 0,
                ..
            }
        ) {
            return Err(invalid(
                "working-set initial size must be strictly positive",
            ));
        }
        Ok(())
    }
}

/// Numerical work performed by a working-set policy for one fitted model.
///
/// Full checks include quadratic subproblem checks in IRLS. Screening has its
/// own diagnostics; a heuristic exclusion is never counted as a certified zero.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkingSetDiagnostics {
    /// Effective policy, with `Auto` resolved for the fitting entry point.
    pub policy: WorkingSetPolicy,
    /// Complete checks of the original problem or a quadratic subproblem.
    pub full_checks: usize,
    /// Checks restricted to the current working set.
    pub restricted_checks: usize,
    /// Changes to the eligible feature set.
    pub rebuilds: usize,
    /// Increases of the adaptive feature budget.
    pub expansions: usize,
    /// Previously excluded features admitted by a subsequent selection.
    pub reactivations: usize,
    /// Features currently eligible for updates, after safe screening.
    pub active_dimension: usize,
    /// Largest eligible dimension during this fit.
    pub peak_active_dimension: usize,
    /// Individual features visited by coordinate sweeps or hybrid PG updates.
    pub feature_visits: usize,
    /// Cluster or segment coordinates visited by hybrid sweeps.
    pub block_visits: usize,
}

pub(crate) struct Workspace {
    pub selected: Vec<bool>,
    pub active: Vec<usize>,
    pub zeroed: Vec<bool>,
    pub scores: Vec<f64>,
    pub diagnostics: WorkingSetDiagnostics,
    history: Vec<bool>,
    order: Vec<usize>,
    budget: usize,
    minimum: usize,
    cold: bool,
    pub initialized: bool,
}

impl Workspace {
    pub fn new(p: usize) -> Self {
        Self {
            selected: vec![true; p],
            active: Vec::with_capacity(p),
            zeroed: vec![false; p],
            scores: vec![0.0; p],
            diagnostics: WorkingSetDiagnostics::default(),
            history: vec![false; p],
            order: Vec::with_capacity(p),
            budget: 0,
            minimum: 0,
            cold: true,
            initialized: false,
        }
    }

    pub fn begin(&mut self, policy: WorkingSetPolicy, path: bool, coefficients: &[f64]) {
        if !path {
            self.history.fill(false);
        }
        self.diagnostics = WorkingSetDiagnostics {
            policy: policy.resolve(path),
            ..Default::default()
        };
        self.begin_model(coefficients);
    }

    pub fn begin_model(&mut self, coefficients: &[f64]) {
        self.record(coefficients);
        self.cold = !self.history.iter().any(|&active| active);
        self.budget = 0;
        self.minimum = 0;
        self.initialized = false;
    }

    pub fn record(&mut self, coefficients: &[f64]) {
        for (active, &value) in self.history.iter_mut().zip(coefficients) {
            *active |= value != 0.0;
        }
    }

    /// Select from a fresh full score vector. Empty screening slices mean no screening.
    pub fn select(
        &mut self,
        coefficients: &[f64],
        screened: &[bool],
        expand: bool,
        stalled: bool,
        tolerance: f64,
    ) -> bool {
        self.record(coefficients);
        let eligible = coefficients
            .iter()
            .enumerate()
            .filter(|&(j, _)| !screened.get(j).copied().unwrap_or(false))
            .count();
        let support = coefficients.iter().filter(|&&b| b != 0.0).count();
        let previous = self.budget;
        let keep_previous = matches!(
            self.diagnostics.policy,
            WorkingSetPolicy::Adaptive { prune: false, .. }
        ) && self.initialized;
        if let WorkingSetPolicy::Adaptive {
            initial_size,
            prune,
        } = self.diagnostics.policy
        {
            let next = if self.initialized && expand {
                if stalled {
                    self.minimum = self.minimum.max(previous).saturating_mul(2).max(1);
                }
                if prune {
                    initial_size.max(support.saturating_mul(2))
                } else {
                    previous.saturating_mul(2)
                }
            } else if self.initialized {
                previous
            } else {
                initial_size
            };
            self.budget = next.max(support).max(self.minimum).min(eligible);
            if self.initialized && self.budget > previous {
                self.diagnostics.expansions += 1;
            }
        }
        self.order.clear();
        // Keep the old membership in scratch until both selection and the
        // reactivation counters have consumed it.
        self.zeroed.copy_from_slice(&self.selected);
        let mut count = 0;
        for (j, &coefficient) in coefficients.iter().enumerate() {
            let allowed = !screened.get(j).copied().unwrap_or(false);
            self.selected[j] = allowed
                && match self.diagnostics.policy {
                    WorkingSetPolicy::Disabled => true,
                    WorkingSetPolicy::EverActive => {
                        self.cold || self.history[j] || self.scores[j] > tolerance
                    }
                    WorkingSetPolicy::Adaptive { .. } => {
                        coefficient != 0.0 || (keep_previous && self.zeroed[j])
                    }
                    WorkingSetPolicy::Auto => unreachable!("the entry point resolves Auto"),
                };
            if self.selected[j] {
                count += 1;
            } else if allowed {
                self.order.push(j);
            }
        }
        if matches!(self.diagnostics.policy, WorkingSetPolicy::Adaptive { .. }) {
            let extra = self.budget.saturating_sub(count).min(self.order.len());
            let compare =
                |&a: &usize, &b: &usize| self.scores[b].total_cmp(&self.scores[a]).then(a.cmp(&b));
            if extra < self.order.len() {
                self.order.select_nth_unstable_by(extra, compare);
            }
            for &j in &self.order[..extra] {
                self.selected[j] = true;
            }
        }
        let mut changed = !self.initialized;
        self.active.clear();
        for j in 0..coefficients.len() {
            changed |= self.selected[j] != self.zeroed[j];
            if self.selected[j] {
                if self.initialized && !self.zeroed[j] {
                    self.diagnostics.reactivations += 1;
                }
                self.active.push(j);
            }
            self.zeroed[j] = !self.selected[j];
        }
        if changed {
            self.diagnostics.rebuilds += 1;
        }
        self.diagnostics.active_dimension = self.active.len();
        self.diagnostics.peak_active_dimension = self
            .diagnostics
            .peak_active_dimension
            .max(self.active.len());
        self.initialized = true;
        self.cold = false;
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_keeps_support_breaks_ties_and_eventually_expands_to_every_feature() {
        for prune in [false, true] {
            let mut work = Workspace::new(8);
            let b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            work.begin(
                WorkingSetPolicy::Adaptive {
                    initial_size: 2,
                    prune,
                },
                false,
                &b,
            );
            work.scores.fill(1.0);
            work.select(&b, &[], false, false, 0.0);
            assert_eq!(work.active, [0, 2]);
            for _ in 0..3 {
                work.select(&b, &[], true, true, 0.0);
            }
            assert_eq!(work.active.len(), 8);
        }
    }

    #[test]
    fn screening_does_not_erase_path_activation_history() {
        let mut work = Workspace::new(3);
        work.begin(WorkingSetPolicy::Auto, true, &[1.0, 0.0, 0.0]);
        work.select(&[0.0; 3], &[true, false, false], false, false, 0.1);
        assert!(work.active.is_empty());
        work.begin(WorkingSetPolicy::Auto, true, &[0.0; 3]);
        work.select(&[0.0; 3], &[], false, false, 0.1);
        assert_eq!(work.active, [0]);
    }

    #[test]
    fn pruning_replaces_zero_members_while_monotone_growth_retains_them() {
        for prune in [false, true] {
            let mut work = Workspace::new(12);
            let mut b = [0.0; 12];
            work.begin(
                WorkingSetPolicy::Adaptive {
                    initial_size: 4,
                    prune,
                },
                false,
                &b,
            );
            work.scores.fill(1.0);
            work.select(&b, &[], false, false, 0.0);
            assert_eq!(work.active, [0, 1, 2, 3]);
            let buffers = (
                work.selected.as_ptr(),
                work.active.as_ptr(),
                work.scores.as_ptr(),
                work.history.as_ptr(),
                work.order.as_ptr(),
                work.zeroed.as_ptr(),
            );
            b[0] = 1.0;
            work.scores[4..].fill(2.0);
            work.select(&b, &[], true, false, 0.0);
            if prune {
                assert_eq!(work.active, [0, 4, 5, 6]);
            } else {
                assert_eq!(work.active, [0, 1, 2, 3, 4, 5, 6, 7]);
            }
            work.begin_model(&b);
            work.select(&b, &[], false, false, 0.0);
            assert_eq!(
                buffers,
                (
                    work.selected.as_ptr(),
                    work.active.as_ptr(),
                    work.scores.as_ptr(),
                    work.history.as_ptr(),
                    work.order.as_ptr(),
                    work.zeroed.as_ptr()
                )
            );
        }
    }
}
