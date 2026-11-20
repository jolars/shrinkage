use extendr_api::Result;
use extendr_api::prelude::*;

fn optional(value: Option<impl Into<Robj>>) -> Robj {
    value.map(Into::into).unwrap_or_else(|| ().into())
}
use faer::MatRef;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use shrinkage::{
    CrossValidation, FoldAssignments, L1, Lasso, LassoFit, Normalization, PathFit, PathOptions,
    PathTermination, RowSubset, SelectionRule, StoppingCriterion, Termination,
};

fn error(message: impl ToString) -> Error {
    Error::Other(message.to_string())
}

fn field(list: &List, name: &str) -> Result<Robj> {
    list.iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| value)
        .ok_or_else(|| error(format!("missing native argument '{name}'")))
}

fn fit_result(fit: &LassoFit, lambda: f64) -> List {
    let preprocessing = fit.preprocessing();
    list!(
        lambda = lambda,
        coefficients = fit.coefficients(),
        intercept = fit.intercept(),
        preprocessing = list!(
            centers = optional(preprocessing.centers()),
            scales = optional(preprocessing.scales())
        ),
        diagnostics = list!(
            termination = match fit.termination() {
                Termination::Converged => "converged",
                Termination::IterationLimit => "iteration_limit",
            },
            iterations = fit.iterations() as f64,
            objective = fit.objective(),
            kkt_violation = fit.kkt_violation(),
            duality_gap = fit.duality_gap(),
            stopping_criterion = match fit.stopping_criterion() {
                StoppingCriterion::DualityGap { .. } => "duality_gap",
                StoppingCriterion::KktViolation { .. } => "kkt_violation",
                StoppingCriterion::ProximalGradientMapping { .. } => "proximal_gradient_mapping",
            },
            stopping_value = fit.stopping_value(),
            stopping_threshold = fit.stopping_threshold()
        )
    )
}

fn lambdas(path: &PathFit<LassoFit, L1>) -> Vec<f64> {
    path.penalties().iter().map(L1::lambda).collect()
}

fn path_result(path: &PathFit<LassoFit, L1>) -> List {
    let fits: List = path
        .iter()
        .map(|(penalty, fit)| fit_result(fit, penalty.lambda()))
        .collect();
    list!(
        fits = fits,
        termination = match path.termination() {
            PathTermination::Completed => "completed",
            PathTermination::DevianceChange => "deviance_change",
            PathTermination::DevianceRatio => "deviance_ratio",
            _ => "unknown",
        },
        lambda_max = optional(path.maximum_penalty().and_then(|maximum| maximum.finite())),
        null_deviance = optional(path.null_deviance()),
        deviances = optional(path.deviances())
    )
}

fn fit<M: RowSubset>(
    x: &M,
    y: &[f64],
    lambda: Option<&[f64]>,
    options: &List,
    cv: Option<List>,
) -> Result<List>
where
    M::Error: std::fmt::Display,
{
    let model = Lasso::new(lambda.and_then(|v| v.first().copied()).unwrap_or(0.0))
        .fit_intercept(field(options, "intercept")?.try_into()?)
        .normalize(if field(options, "standardize")?.try_into()? {
            Normalization::Auto
        } else {
            Normalization::None
        })
        .max_iterations(field(options, "max_iterations")?.try_into()?);
    let tolerance = field(options, "tolerance")?.as_real();
    let model = match tolerance {
        Some(tolerance) => model.tolerance(tolerance),
        None => model,
    };
    let path = PathOptions::new()
        .nlambda(field(options, "nlambda")?.try_into()?)
        .early_stopping(field(options, "early_stopping")?.try_into()?);
    let ratio = field(options, "lambda_min_ratio")?.as_real();
    let path = match ratio {
        Some(ratio) => path.lambda_min_ratio(ratio),
        None => path,
    };

    if let Some(cv) = cv {
        let ids = field(&cv, "foldid")?;
        let ids = ids.as_integer_slice();
        let folds = match ids {
            Some(ids) => FoldAssignments::new(
                ids.iter()
                    .map(|&id| {
                        id.checked_sub(1)
                            .and_then(|id| usize::try_from(id).ok())
                            .ok_or_else(|| error("invalid fold ID"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => FoldAssignments::kfold(
                y.len(),
                field(&cv, "nfolds")?.try_into()?,
                field(&cv, "seed")?.try_into()?,
            ),
        }
        .map_err(error)?;
        let one_se: bool = field(&cv, "one_se")?.try_into()?;
        let cv = CrossValidation::new(folds).selection_rule(if one_se {
            SelectionRule::OneStandardError
        } else {
            SelectionRule::MinimumError
        });
        let result = match lambda {
            Some(lambda) => model.cross_validate_path(x, y, lambda, &cv),
            None => model.cross_validate_regularization_path(x, y, &path, &cv),
        }
        .map_err(error)?;
        let fold_scores: List = result
            .folds()
            .iter()
            .map(|fold| fold.scores().into_robj())
            .collect();
        let fold_lambdas: List = result
            .folds()
            .iter()
            .map(|fold| lambdas(fold.path()))
            .collect();
        return Ok(list!(
            path = path_result(result.refit_path()),
            foldid = result
                .assignments()
                .assignments()
                .iter()
                .map(|&id| id as i32 + 1)
                .collect::<Vec<_>>(),
            fold_scores = fold_scores,
            fold_lambdas = fold_lambdas,
            mean_scores = result.mean_scores(),
            standard_errors = result.standard_errors(),
            relative_strengths = optional(result.relative_strengths()),
            minimum_index = result.minimum_index() as i32 + 1,
            one_se_index = result.one_standard_error_index() as i32 + 1,
            selected_index = result.selected_index() as i32 + 1
        ));
    }

    match lambda {
        Some([lambda]) => model
            .fit(x, y)
            .map(|fit| fit_result(&fit, *lambda))
            .map_err(error),
        Some(lambda) => model
            .fit_path(x, y, lambda)
            .map(|path| path_result(&path))
            .map_err(error),
        None => model
            .fit_regularization_path(x, y, &path)
            .map(|path| path_result(&path))
            .map_err(error),
    }
}

#[extendr]
fn shrinkage_fit_native(
    design: List,
    y: &[f64],
    lambda: Robj,
    options: List,
    cv: Robj,
) -> Result<List> {
    let cv = if cv.is_null() {
        None
    } else {
        Some(cv.try_into()?)
    };
    let dims = field(&design, "dim")?;
    let [rows, cols] = dims.as_integer_slice().unwrap_or_default() else {
        return Err(error("expected a two-dimensional design"));
    };
    let rows = usize::try_from(*rows).map_err(error)?;
    let cols = usize::try_from(*cols).map_err(error)?;
    let values = field(&design, "values")?;
    let values = values
        .as_real_slice()
        .ok_or_else(|| error("expected double matrix values"))?;
    let lambda = if lambda.is_null() {
        None
    } else {
        Some(
            lambda
                .as_real_slice()
                .ok_or_else(|| error("expected double lambda values"))?,
        )
    };

    let offsets = field(&design, "offsets")?;
    if offsets.is_null() {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(error("invalid dense matrix dimensions"));
        }
        let x = MatRef::from_column_major_slice(values, rows, cols).as_dyn();
        return fit(&x, y, lambda, &options, cv);
    }

    let indices = field(&design, "indices")?;
    let as_indices = |value: &Robj| -> Result<Vec<usize>> {
        value
            .as_integer_slice()
            .ok_or_else(|| error("invalid CSC indices"))?
            .iter()
            .map(|&index| usize::try_from(index).map_err(error))
            .collect()
    };
    let offsets = as_indices(&offsets)?;
    let indices = as_indices(&indices)?;
    if offsets.len() != cols + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&values.len())
        || indices.len() != values.len()
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || indices.iter().any(|&index| index >= rows)
        || offsets.windows(2).any(|pair| {
            indices[pair[0]..pair[1]]
                .windows(2)
                .any(|indices| indices[0] >= indices[1])
        })
    {
        return Err(error("invalid dgCMatrix CSC storage"));
    }

    let symbolic = SymbolicSparseColMatRef::new_checked(rows, cols, &offsets, None, &indices);
    let x = SparseColMatRef::new(symbolic, values);
    fit(&x, y, lambda, &options, cv)
}

extendr_module! {
    mod shrinkage;
    fn shrinkage_fit_native;
}
