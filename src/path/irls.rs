use lazymatrix::{ColumnStats, LazyMatrix, RawColumns};

use crate::error::{finite, invalid};
use crate::solver::irls::{Workspace, iterate, validate_rows};
use crate::{
    CoordinatePenalty, DiagonalHessianDatafit, FitError, Irls, IrlsFit, NullModelDatafit, PathFit,
    PathOptions, PathPenalty, PathSolver, PenaltyMaximum, Problem, RegularizationPathSolver,
};

impl<M, D, P, Q> PathSolver<Problem<&M, D, P>, Q> for Irls
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
    D: DiagonalHessianDatafit,
    Q: CoordinatePenalty,
{
    type Fit = IrlsFit;
    type Error = FitError<M::Error>;

    fn solve_path(
        &self,
        problem: &Problem<&M, D, P>,
        penalties: Vec<Q>,
    ) -> Result<PathFit<IrlsFit, Q>, Self::Error> {
        self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let zeros = vec![0.0; problem.design().ncols()];
        problem
            .datafit()
            .value(&vec![0.0; problem.design().nrows()])
            .map_err(|e| FitError::from_component(e, 0))?;
        for penalty in &penalties {
            penalty
                .validate_input(&zeros)
                .map_err(|e| FitError::from_component(e, 0))?;
        }
        if penalties.is_empty() {
            return Ok(PathFit::explicit(Vec::new(), penalties));
        }
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut work = Workspace::new(prepared.matrix.nrows(), prepared.matrix.ncols());
        work.path = true;
        let solutions = penalties
            .iter()
            .map(|penalty| {
                iterate(
                    &prepared.matrix,
                    problem.datafit(),
                    penalty,
                    self,
                    &mut work,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let preprocessing = prepared.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|solution| IrlsFit::from_solution(solution, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::explicit(fits, penalties))
    }
}

impl<M, D, P> RegularizationPathSolver<Problem<&M, D, P>> for Irls
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
    D: DiagonalHessianDatafit + NullModelDatafit,
    P: PathPenalty,
    P::AtStrength: CoordinatePenalty,
{
    type Fit = IrlsFit;
    type Penalty = P::AtStrength;
    type Error = FitError<M::Error>;

    fn maximum_penalty(&self, problem: &Problem<&M, D, P>) -> Result<PenaltyMaximum, Self::Error> {
        self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let (_, gradient) = null_state(&prepared.matrix, problem.datafit())?;
        problem
            .penalty()
            .maximum_penalty(&gradient)
            .map_err(|e| FitError::from_component(e, 0))
    }

    fn solve_regularization_path(
        &self,
        problem: &Problem<&M, D, P>,
        options: &PathOptions,
    ) -> Result<PathFit<IrlsFit, P::AtStrength>, Self::Error> {
        options.validate()?;
        self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let (intercept, gradient) = null_state(&prepared.matrix, problem.datafit())?;
        let mut sequence = options
            .sequence(
                problem.penalty(),
                &gradient,
                prepared.matrix.nrows(),
                prepared.matrix.ncols(),
            )
            .map_err(|e| FitError::from_component(e, 0))?;
        let mut work = Workspace::new(prepared.matrix.nrows(), prepared.matrix.ncols());
        work.path = true;
        work.intercept = intercept;
        let null = super::null_deviance(problem.datafit(), prepared.matrix.nrows())?;
        let solutions = sequence.fit(options, null, |penalty| {
            let solution = iterate(
                &prepared.matrix,
                problem.datafit(),
                penalty,
                self,
                &mut work,
            )?;
            let deviance = problem
                .datafit()
                .deviance(&work.eta)
                .map_err(|e| FitError::from_component(e, 0))?;
            let termination = solution.termination;
            Ok((solution, deviance, termination))
        })?;
        let preprocessing = prepared.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|solution| IrlsFit::from_solution(solution, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::automatic(fits, sequence))
    }
}

fn null_state<M, D, E>(matrix: &LazyMatrix<M>, datafit: &D) -> Result<(f64, Vec<f64>), FitError<E>>
where
    M: RawColumns<f64>,
    D: DiagonalHessianDatafit + NullModelDatafit,
{
    let intercepts = datafit
        .null_intercepts()
        .map_err(|e| FitError::from_component(e, 0))?;
    if intercepts.len() != 1
        || !intercepts[0].is_finite()
        || (!datafit.fits_intercept() && intercepts[0] != 0.0)
    {
        return Err(invalid(
            "a vector datafit must return one finite null intercept, zero when disabled",
        ));
    }
    let mut derivative = vec![0.0; matrix.nrows()];
    datafit
        .gradient(&vec![intercepts[0]; matrix.nrows()], &mut derivative)
        .map_err(|e| FitError::from_component(e, 0))?;
    for &value in &derivative {
        finite(value, "checking the null gradient", 0)?;
    }
    let sum = finite(derivative.iter().sum(), "summing the null gradient", 0)?;
    let gradient = (0..matrix.ncols())
        .map(|j| {
            finite(
                matrix.column(j).dot_with_sum(&derivative, sum),
                "computing the null coefficient gradient",
                0,
            )
            .map_err(FitError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((intercepts[0], gradient))
}
