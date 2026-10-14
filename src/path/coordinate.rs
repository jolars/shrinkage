use lazymatrix::{ColumnStats, RawColumns};

use crate::error::finite;
use crate::solver::coordinate::{Workspace, solve_with_workspace};
use crate::{
    CoordinateDescent, FitError, Gaussian, L1, LassoFit, NullModelDatafit, PathFit, PathOptions,
    PathPenalty, PathSolver, PenaltyMaximum, Problem, RegularizationPathSolver,
};

impl<M, P> PathSolver<Problem<&M, Gaussian<'_>, P>, L1> for CoordinateDescent
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
{
    type Fit = LassoFit;
    type Error = FitError<M::Error>;
    fn solve_path(
        &self,
        problem: &Problem<&M, Gaussian<'_>, P>,
        penalties: Vec<L1>,
    ) -> Result<PathFit<LassoFit, L1>, Self::Error> {
        self.validate(self.criterion(L1::new(1.0)))?;
        problem.datafit().validate(problem.design().nrows())?;
        for &penalty in &penalties {
            penalty.validate()?;
            self.validate(self.criterion(penalty))?;
        }
        if penalties.is_empty() {
            return Ok(PathFit::explicit(Vec::new(), penalties));
        }
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut work = Workspace::new(&prepared.matrix, *problem.datafit())?;
        let solutions = penalties
            .iter()
            .map(|&penalty| {
                solve_with_workspace(
                    &prepared.matrix,
                    *problem.datafit(),
                    penalty,
                    self,
                    &mut work,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let preprocessing = prepared.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|solution| LassoFit::from_solution(solution, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::explicit(fits, penalties))
    }
}

impl<M> RegularizationPathSolver<Problem<&M, Gaussian<'_>, L1>> for CoordinateDescent
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
{
    type Fit = LassoFit;
    type Penalty = L1;
    type Error = FitError<M::Error>;
    fn maximum_penalty(
        &self,
        problem: &Problem<&M, Gaussian<'_>, L1>,
    ) -> Result<PenaltyMaximum, Self::Error> {
        self.validate(self.criterion(*problem.penalty()))?;
        problem.datafit().validate(problem.design().nrows())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let gradient = null_gradient(&prepared.matrix, *problem.datafit())?;
        problem
            .penalty()
            .maximum_penalty(&gradient)
            .map_err(|e| FitError::from_component(e, 0))
    }
    fn solve_regularization_path(
        &self,
        problem: &Problem<&M, Gaussian<'_>, L1>,
        options: &PathOptions,
    ) -> Result<PathFit<LassoFit, L1>, Self::Error> {
        options.validate()?;
        self.validate(self.criterion(*problem.penalty()))?;
        problem.datafit().validate(problem.design().nrows())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let gradient = null_gradient(&prepared.matrix, *problem.datafit())?;
        let sequence = options
            .sequence(
                problem.penalty(),
                &gradient,
                prepared.matrix.nrows(),
                prepared.matrix.ncols(),
            )
            .map_err(|e| FitError::from_component(e, 0))?;
        let mut work = Workspace::new(&prepared.matrix, *problem.datafit())?;
        let solutions = sequence
            .penalties
            .iter()
            .map(|&penalty| {
                solve_with_workspace(
                    &prepared.matrix,
                    *problem.datafit(),
                    penalty,
                    self,
                    &mut work,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let preprocessing = prepared.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|solution| LassoFit::from_solution(solution, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::automatic(fits, sequence))
    }
}

fn null_gradient<M: RawColumns<f64>, E>(
    matrix: &lazymatrix::LazyMatrix<M>,
    datafit: Gaussian<'_>,
) -> Result<Vec<f64>, FitError<E>> {
    let intercept = datafit
        .null_intercepts()
        .map_err(|e| FitError::from_component(e, 0))?[0];
    let n = matrix.nrows() as f64;
    let derivative = datafit
        .response()
        .iter()
        .map(|y| finite((intercept - y) / n, "computing the null gradient", 0))
        .collect::<Result<Vec<_>, _>>()?;
    let sum = finite(derivative.iter().sum(), "summing the null gradient", 0)?;
    (0..matrix.ncols())
        .map(|j| {
            Ok(finite(
                matrix.column(j).dot_with_sum(&derivative, sum),
                "computing the null coefficient gradient",
                0,
            )?)
        })
        .collect()
}

impl<D, P, Q> PathSolver<Problem<Box<dyn crate::RuntimeDesign + '_>, D, P>, Q> for CoordinateDescent
where
    D: crate::SmoothDatafit,
    Q: crate::ProximalPenalty,
{
    type Fit = LassoFit;
    type Error = FitError<crate::BackendError>;
    fn solve_path(
        &self,
        _problem: &Problem<Box<dyn crate::RuntimeDesign + '_>, D, P>,
        _penalties: Vec<Q>,
    ) -> Result<PathFit<LassoFit, Q>, Self::Error> {
        Err(runtime_unsupported())
    }
}

impl<D, P> RegularizationPathSolver<Problem<Box<dyn crate::RuntimeDesign + '_>, D, P>>
    for CoordinateDescent
where
    D: crate::SmoothDatafit + NullModelDatafit,
    P: PathPenalty,
{
    type Fit = LassoFit;
    type Penalty = P::AtStrength;
    type Error = FitError<crate::BackendError>;
    fn maximum_penalty(
        &self,
        _problem: &Problem<Box<dyn crate::RuntimeDesign + '_>, D, P>,
    ) -> Result<PenaltyMaximum, Self::Error> {
        Err(runtime_unsupported())
    }
    fn solve_regularization_path(
        &self,
        _problem: &Problem<Box<dyn crate::RuntimeDesign + '_>, D, P>,
        _options: &PathOptions,
    ) -> Result<PathFit<LassoFit, P::AtStrength>, Self::Error> {
        Err(runtime_unsupported())
    }
}

fn runtime_unsupported<E>() -> FitError<E> {
    crate::error::unsupported(
        "CoordinateDescent",
        "runtime proximal oracles do not expose Gaussian/L1 coordinate updates or a dual certificate",
        "use ProximalGradient<Vec<f64>> for runtime paths, or a typed Gaussian/L1 problem",
    )
}
