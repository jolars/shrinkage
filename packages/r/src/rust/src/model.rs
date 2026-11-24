use extendr_api::Result;
use extendr_api::prelude::*;
use faer::Col;
use shrinkage_core::lazymatrix::{MatTransposeVecInto, MatVecInto};
use shrinkage_core::{
    Binomial, CoefficientLayout, CoordinateDescent, CrossValidation, CrossValidationDatafit,
    FitError, FoldAssignments, Gaussian, HybridCoordinateDescent, Irls, MultivariateGaussian,
    Normalization, PathOptions, PathPenalty, PathSolver, Probit, Problem, ProximalGradient,
    RegularizationPathSolver, RowSubset, SelectionRule, Solver, ValidationFit, ValidationMetric,
};

use crate::penalty::{self, Proximal, string};
use crate::result::{FitRecord, PenaltyRecord, cv_result, path_result};
use crate::{error, field};

#[derive(Clone, Copy, PartialEq)]
enum Loss {
    Gaussian,
    MultivariateGaussian,
    Logit,
    Probit,
}

enum Operation {
    Single,
    Explicit,
    Automatic(PathOptions),
}

pub(crate) struct Request {
    loss: Loss,
    operation: Operation,
    penalties: Vec<List>,
    cv: Option<CrossValidation>,
    solver: String,
    intercept: bool,
    nresponses: usize,
    normalization: Normalization,
    tolerance: Option<f64>,
    max_iterations: usize,
}

fn cv_options(y: &[f64], cv: List, loss: Loss) -> Result<CrossValidation> {
    if loss == Loss::Probit {
        return Err(error("cross-validation is unavailable for the probit link"));
    }
    if loss == Loss::MultivariateGaussian {
        return Err(error(
            "cross-validation is unavailable for multivariate losses",
        ));
    }
    let ids = field(&cv, "foldid")?;
    let folds = if let Some(ids) = ids.as_integer_slice() {
        FoldAssignments::new(
            ids.iter()
                .map(|&id| {
                    id.checked_sub(1)
                        .and_then(|id| usize::try_from(id).ok())
                        .ok_or_else(|| error("invalid fold ID"))
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        let nfolds = field(&cv, "nfolds")?.try_into()?;
        let seed = field(&cv, "seed")?.try_into()?;
        if field(&cv, "stratified")?.try_into()? {
            if loss != Loss::Logit {
                return Err(error("stratified folds require a binomial loss"));
            }
            FoldAssignments::stratified_binomial(y, nfolds, seed)
        } else {
            FoldAssignments::kfold(y.len(), nfolds, seed)
        }
    }
    .map_err(error)?;
    let cv_config =
        CrossValidation::new(folds).selection_rule(match string(&cv, "rule")?.as_str() {
            "minimum" => SelectionRule::MinimumError,
            "one_se" => SelectionRule::OneStandardError,
            _ => return Err(error("unknown cross-validation selection rule")),
        });
    let metric = match string(&cv, "metric")?.as_str() {
        "auto" => return Ok(cv_config),
        "mse" => ValidationMetric::MeanSquaredError,
        "mae" => ValidationMetric::MeanAbsoluteError,
        "log_loss" => ValidationMetric::LogLoss,
        "deviance" => ValidationMetric::BinomialDeviance,
        _ => return Err(error("unknown cross-validation metric")),
    };
    Ok(cv_config.metric(metric))
}

// Each solver is built with its own default stopping criterion.
macro_rules! configure_solver {
    ($request:expr, $solver:expr) => {{
        let solver = $solver.max_iterations($request.max_iterations);
        match $request.tolerance {
            Some(tolerance) => solver.tolerance(tolerance),
            None => solver,
        }
    }};
}

impl Request {
    pub(crate) fn new(
        y: &[f64],
        loss: &List,
        penalty: &List,
        options: &List,
        cv: Robj,
    ) -> Result<Self> {
        let loss = match (
            string(loss, "name")?.as_str(),
            string(loss, "link")?.as_str(),
        ) {
            ("gaussian", "identity") => Loss::Gaussian,
            ("multivariate_gaussian", "identity") => Loss::MultivariateGaussian,
            ("binomial", "logit") => Loss::Logit,
            ("binomial", "probit") => Loss::Probit,
            _ => return Err(error("unsupported loss and link combination")),
        };
        let operation = match string(penalty, "name")?.as_str() {
            "single" => Operation::Single,
            "explicit" => Operation::Explicit,
            "auto" => {
                let mut path = PathOptions::new()
                    .nlambda(field(penalty, "nlambda")?.try_into()?)
                    .early_stopping(field(penalty, "early_stopping")?.try_into()?);
                if let Some(ratio) = field(penalty, "lambda_min_ratio")?.as_real() {
                    path = path.lambda_min_ratio(ratio);
                }
                if let Some(start) = field(penalty, "lambda_start")?.as_real() {
                    path = path.lambda_start(start);
                }
                Operation::Automatic(path)
            }
            _ => return Err(error("unknown path specification")),
        };
        let penalties = List::try_from(field(penalty, "penalties")?)?
            .values()
            .map(List::try_from)
            .collect::<Result<Vec<_>>>()?;
        if penalties.is_empty()
            || (!matches!(operation, Operation::Explicit) && penalties.len() != 1)
        {
            return Err(error("invalid number of penalties"));
        }
        let normalization = match string(options, "normalization")?.as_str() {
            "auto" => Normalization::Auto,
            "none" => Normalization::None,
            "standardize" => Normalization::Standardize,
            "center" => Normalization::Center,
            "min_max" => Normalization::MinMax,
            "max_abs" => Normalization::MaxAbs,
            "l1" => Normalization::L1,
            "l2" => Normalization::L2,
            _ => return Err(error("unknown normalization")),
        };
        Ok(Self {
            loss,
            operation,
            penalties,
            cv: if cv.is_null() {
                None
            } else {
                Some(cv_options(y, cv.try_into()?, loss)?)
            },
            solver: string(options, "solver")?,
            intercept: field(options, "intercept")?.try_into()?,
            nresponses: field(options, "nresponses")?.try_into()?,
            normalization,
            tolerance: field(options, "tolerance")?.as_real(),
            max_iterations: field(options, "max_iterations")?.try_into()?,
        })
    }

    pub(crate) fn fit<M>(&self, x: &M, y: &[f64]) -> Result<List>
    where
        M: RowSubset + MatVecInto<Col<f64>> + MatTransposeVecInto<Col<f64>>,
        M::Subset: MatVecInto<Col<f64>> + MatTransposeVecInto<Col<f64>>,
        M::Error: std::fmt::Display,
    {
        let name = string(&self.penalties[0], "name")?;
        let mixed = self.penalties[1..]
            .iter()
            .map(|penalty| string(penalty, "name"))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .any(|other| *other != name);
        let solver = if self.solver == "auto" {
            match (self.loss, name.as_str(), mixed) {
                (Loss::Gaussian, "l1", false) => "coordinate_descent",
                (
                    Loss::Gaussian | Loss::Logit | Loss::Probit,
                    "l1" | "ridge" | "elastic_net",
                    false,
                ) => "irls",
                (
                    Loss::Gaussian | Loss::Logit,
                    "clustered_lasso" | "chain_fused_lasso" | "slope",
                    false,
                ) => "hybrid_coordinate_descent",
                _ => "proximal_gradient",
            }
        } else {
            self.solver.as_str()
        };
        let mismatch = || {
            error(format!(
                "solver '{solver}' does not support this loss and penalty combination"
            ))
        };
        if self.loss == Loss::MultivariateGaussian {
            if solver != "proximal_gradient" {
                return Err(mismatch());
            }
            let layout = CoefficientLayout::new(x.ncols(), self.nresponses).map_err(error)?;
            let datafit = MultivariateGaussian::new(y, x.nrows(), self.nresponses)
                .fit_intercept(self.intercept);
            let points = self.points(|spec| penalty::axis(spec, layout))?;
            let solver = configure_solver!(self, ProximalGradient::<Col<f64>>::new());
            return self.solve_without_cv(x, datafit, points, &solver);
        }
        if mixed && solver != "proximal_gradient" {
            return Err(error(format!(
                "solver '{solver}' requires one penalty type per path"
            )));
        }
        let gaussian = Gaussian::new(y).fit_intercept(self.intercept);
        let binomial = Binomial::new(y).fit_intercept(self.intercept);
        let coordinate = || configure_solver!(self, CoordinateDescent::new());
        let irls = || configure_solver!(self, Irls::new());
        let proximal = || configure_solver!(self, ProximalGradient::<Col<f64>>::new());
        let hybrid = || configure_solver!(self, HybridCoordinateDescent::<Col<f64>>::new());

        macro_rules! canonical {
            ($solver:expr, $points:expr) => {
                match self.loss {
                    Loss::Gaussian => self.solve(x, gaussian, $points, &$solver),
                    Loss::Logit => self.solve(x, binomial, $points, &$solver),
                    _ => Err(mismatch()),
                }
            };
        }
        macro_rules! coordinate_penalty {
            ($points:expr) => {
                match solver {
                    "irls" if self.loss == Loss::Probit => {
                        self.solve_without_cv(x, binomial.with_link(Probit), $points, &irls())
                    }
                    "irls" => canonical!(irls(), $points),
                    _ => Err(mismatch()),
                }
            };
        }
        macro_rules! structured_penalty {
            ($points:expr) => {
                match solver {
                    "hybrid_coordinate_descent" => canonical!(hybrid(), $points),
                    _ => Err(mismatch()),
                }
            };
        }

        if solver == "proximal_gradient" {
            return canonical!(
                proximal(),
                self.points(|spec| Proximal::parse(spec, x.ncols()))?
            );
        }
        match name.as_str() {
            "l1" => {
                let points = self.points(penalty::l1)?;
                if solver == "coordinate_descent" && self.loss == Loss::Gaussian {
                    self.solve(x, gaussian, points, &coordinate())
                } else {
                    coordinate_penalty!(points)
                }
            }
            "ridge" => coordinate_penalty!(self.points(penalty::ridge)?),
            "elastic_net" => coordinate_penalty!(self.points(penalty::elastic_net)?),
            "clustered_lasso" => structured_penalty!(self.points(penalty::clustered_lasso)?),
            "chain_fused_lasso" => structured_penalty!(self.points(penalty::chain_fused_lasso)?),
            "slope" => structured_penalty!(self.points(penalty::slope)?),
            _ => Err(mismatch()),
        }
    }

    fn points<P>(&self, convert: impl FnMut(&List) -> Result<P>) -> Result<Vec<P>> {
        self.penalties.iter().map(convert).collect()
    }

    fn solve<'m, M, D, P, F, S>(
        &self,
        x: &'m M,
        datafit: D,
        points: Vec<P>,
        solver: &S,
    ) -> Result<List>
    where
        M: RowSubset,
        M::Error: std::fmt::Display,
        D: CrossValidationDatafit,
        P: PathPenalty<AtStrength = P> + PenaltyRecord + Clone,
        F: FitRecord + ValidationFit,
        S: Solver<Problem<&'m M, D, P>, Fit = F, Error = FitError<M::Error>>
            + PathSolver<Problem<&'m M, D, P>, P, Fit = F, Error = FitError<M::Error>>
            + RegularizationPathSolver<
                Problem<&'m M, D, P>,
                Fit = F,
                Penalty = P,
                Error = FitError<M::Error>,
            >,
        for<'a> S: PathSolver<
                Problem<&'a M::Subset, D::WithResponse<'a>, ()>,
                P,
                Fit = F,
                Error = FitError<M::Error>,
            > + RegularizationPathSolver<
                Problem<&'a M::Subset, D::WithResponse<'a>, P>,
                Fit = F,
                Penalty = P,
                Error = FitError<M::Error>,
            >,
    {
        let Some(cv) = &self.cv else {
            return self.solve_without_cv(x, datafit, points, solver);
        };
        let problem = Problem::new(x, datafit, points[0].clone()).normalize(self.normalization);
        let result = match self.operation {
            Operation::Automatic(options) => {
                problem.cross_validate_regularization_path_with(solver, &options, cv)
            }
            Operation::Single | Operation::Explicit => {
                problem.cross_validate_path_with(solver, points, cv)
            }
        }
        .map_err(error)?;
        Ok(cv_result(&result))
    }

    fn solve_without_cv<'m, M, D, P, F, S>(
        &self,
        x: &'m M,
        datafit: D,
        points: Vec<P>,
        solver: &S,
    ) -> Result<List>
    where
        P: PenaltyRecord + Clone,
        F: FitRecord,
        S: Solver<Problem<&'m M, D, P>, Fit = F>
            + PathSolver<Problem<&'m M, D, P>, P, Fit = F>
            + RegularizationPathSolver<Problem<&'m M, D, P>, Fit = F, Penalty = P>,
        <S as Solver<Problem<&'m M, D, P>>>::Error: std::fmt::Display,
        <S as PathSolver<Problem<&'m M, D, P>, P>>::Error: std::fmt::Display,
        <S as RegularizationPathSolver<Problem<&'m M, D, P>>>::Error: std::fmt::Display,
    {
        let problem = Problem::new(x, datafit, points[0].clone()).normalize(self.normalization);
        match self.operation {
            Operation::Single => {
                let fit = problem.fit_with(solver).map_err(error)?;
                Ok(fit.record(problem.penalty().record(), problem.penalty().lambda()))
            }
            Operation::Explicit => {
                let path = problem.fit_path_with(solver, points).map_err(error)?;
                Ok(path_result(&path))
            }
            Operation::Automatic(options) => {
                let path = problem
                    .fit_regularization_path_with(solver, &options)
                    .map_err(error)?;
                Ok(path_result(&path))
            }
        }
    }
}
