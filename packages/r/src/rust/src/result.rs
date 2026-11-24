use extendr_api::prelude::*;
use shrinkage_core::{
    AxisPenalty, ChainFusedLasso, ClusteredLasso, CrossValidationFit, ElasticNet, HybridFit,
    IrlsFit, L1, LassoFit, MultivariateFit, PathFit, PathTermination, PenaltyAxis, ProximalFit,
    Ridge, SelectionRule, Slope, StoppingCriterion, Termination, ValidationMetric,
};

use crate::optional;

pub(crate) trait PenaltyRecord {
    fn record(&self) -> List;
    fn lambda(&self) -> Option<f64> {
        None
    }
}

macro_rules! scalar_penalty {
    ($type:ty, $name:literal) => {
        impl PenaltyRecord for $type {
            fn record(&self) -> List {
                list!(name = $name, lambda = self.lambda())
            }

            fn lambda(&self) -> Option<f64> {
                Some(<$type>::lambda(self))
            }
        }
    };
}

scalar_penalty!(L1, "l1");
scalar_penalty!(Ridge, "ridge");

impl PenaltyRecord for ElasticNet {
    fn record(&self) -> List {
        list!(name = "elastic_net", l1 = self.l1(), l2 = self.l2())
    }
}

impl PenaltyRecord for ClusteredLasso {
    fn record(&self) -> List {
        list!(
            name = "clustered_lasso",
            l1 = self.l1(),
            fusion = self.fusion()
        )
    }
}

impl PenaltyRecord for ChainFusedLasso {
    fn record(&self) -> List {
        list!(
            name = "chain_fused_lasso",
            l1 = self.l1(),
            fusion = self.fusion(),
            weights = optional(self.weights())
        )
    }
}

impl PenaltyRecord for Slope {
    fn record(&self) -> List {
        list!(name = "slope", weights = self.weights())
    }
}

pub(crate) trait FitRecord {
    fn record(&self, penalty: List, lambda: Option<f64>) -> List;
}

macro_rules! fit_record {
    ($type:ty, $solver:literal, $intercept:ident, $( $diagnostic:ident : $value:expr ),* $(,)?) => {
        impl FitRecord for $type {
            fn record(&self, penalty: List, lambda: Option<f64>) -> List {
                let preprocessing = self.preprocessing();
                list!(
                    penalty = penalty,
                    lambda = optional(lambda),
                    solver = $solver,
                    coefficients = self.coefficients(),
                    intercept = self.$intercept(),
                    preprocessing = list!(
                        centers = optional(preprocessing.centers()),
                        scales = optional(preprocessing.scales())
                    ),
                    diagnostics = list!(
                        termination = match self.termination() {
                            Termination::Converged => "converged",
                            Termination::IterationLimit => "iteration_limit",
                        },
                        iterations = self.iterations() as f64,
                        objective = self.objective(),
                        stopping_criterion = match self.stopping_criterion() {
                            StoppingCriterion::DualityGap { .. } => "duality_gap",
                            StoppingCriterion::KktViolation { .. } => "kkt_violation",
                            StoppingCriterion::ProximalGradientMapping { .. } => "proximal_gradient_mapping",
                        },
                        stopping_value = self.stopping_value(),
                        stopping_threshold = self.stopping_threshold()
                        $(, $diagnostic = $value(self))*
                    )
                )
            }
        }
    };
}

fit_record!(LassoFit, "coordinate_descent", intercept, kkt_violation: LassoFit::kkt_violation,
            duality_gap: LassoFit::duality_gap);
fit_record!(IrlsFit, "irls", intercept, kkt_violation: IrlsFit::kkt_violation);
fit_record!(HybridFit, "hybrid_coordinate_descent", intercept, duality_gap: (|fit: &HybridFit| optional(fit.duality_gap())));
fit_record!(ProximalFit, "proximal_gradient", intercept, duality_gap: (|fit: &ProximalFit| optional(fit.duality_gap())));

fit_record!(MultivariateFit, "proximal_gradient", intercepts,);

impl<P: PenaltyRecord> PenaltyRecord for AxisPenalty<P> {
    fn record(&self) -> List {
        list!(
            name = "axis",
            axis = match self.axis() {
                PenaltyAxis::All => "all",
                PenaltyAxis::Features => "features",
                PenaltyAxis::Responses => "responses",
            },
            penalty = self.penalty().record()
        )
    }

    fn lambda(&self) -> Option<f64> {
        self.penalty().lambda()
    }
}

fn lambdas<F, P: PenaltyRecord>(path: &PathFit<F, P>) -> Option<Vec<f64>> {
    path.lambdas()
        .map(<[f64]>::to_vec)
        .or_else(|| path.penalties().iter().map(PenaltyRecord::lambda).collect())
}

pub(crate) fn path_result<F: FitRecord, P: PenaltyRecord>(path: &PathFit<F, P>) -> List {
    let lambda = lambdas(path);
    let fits: List = path
        .iter()
        .enumerate()
        .map(|(i, (penalty, fit))| {
            fit.record(penalty.record(), lambda.as_ref().map(|values| values[i]))
        })
        .collect();
    list!(
        fits = fits,
        lambda = optional(lambda),
        termination = match path.termination() {
            PathTermination::Completed => "completed",
            PathTermination::DevianceChange => "deviance_change",
            PathTermination::DevianceRatio => "deviance_ratio",
            _ => "unknown",
        },
        lambda_max = optional(path.maximum_penalty().and_then(|maximum| maximum.finite())),
        lambda_start = optional(path.lambda_start()),
        null_deviance = optional(path.null_deviance()),
        deviances = optional(path.deviances())
    )
}

pub(crate) fn cv_result<F: FitRecord, P: PenaltyRecord>(result: &CrossValidationFit<F, P>) -> List {
    let fold_scores: List = result
        .folds()
        .iter()
        .map(|fold| fold.scores().into_robj())
        .collect();
    let fold_lambdas = result
        .folds()
        .iter()
        .map(|fold| lambdas(fold.path()))
        .collect::<Option<List>>();
    list!(
        path = path_result(result.refit_path()),
        foldid = result
            .assignments()
            .assignments()
            .iter()
            .map(|&id| id as i32 + 1)
            .collect::<Vec<_>>(),
        fold_scores = fold_scores,
        fold_lambdas = optional(fold_lambdas),
        mean_scores = result.mean_scores(),
        standard_errors = result.standard_errors(),
        relative_strengths = optional(result.relative_strengths()),
        metric = match result.metric() {
            ValidationMetric::MeanSquaredError => "mse",
            ValidationMetric::MeanAbsoluteError => "mae",
            ValidationMetric::LogLoss => "log_loss",
            ValidationMetric::BinomialDeviance => "deviance",
        },
        rule = match result.selection_rule() {
            SelectionRule::MinimumError => "minimum",
            SelectionRule::OneStandardError => "one_se",
        },
        minimum_index = result.minimum_index() as i32 + 1,
        one_se_index = result.one_standard_error_index() as i32 + 1,
        selected_index = result.selected_index() as i32 + 1
    )
}
