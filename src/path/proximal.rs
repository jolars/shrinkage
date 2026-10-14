use lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, RawColumns};

use crate::error::{finite, invalid};
use crate::solver::proximal::{NativeDesign, Workspace, iterate_path, iterate_with_workspace};
use crate::{
    BackendError, FitError, NullModelDatafit, PathFit, PathOptions, PathPenalty, PathSolver,
    PenaltyMaximum, Problem, ProximalDesign, ProximalFit, ProximalGradient, ProximalPenalty,
    ProximalVector, RegularizationPathSolver, RuntimeDesign, SmoothDatafit,
};

impl<M, D, P, Q, V> PathSolver<Problem<&M, D, P>, Q> for ProximalGradient<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    D: SmoothDatafit,
    Q: ProximalPenalty,
    V: ProximalVector,
{
    type Fit = ProximalFit;
    type Error = FitError<M::Error>;
    fn solve_path(
        &self,
        problem: &Problem<&M, D, P>,
        penalties: Vec<Q>,
    ) -> Result<PathFit<ProximalFit, Q>, Self::Error> {
        let fits = self.solve_path(problem, &penalties)?;
        Ok(PathFit::explicit(fits, penalties))
    }
}

impl<M, D, P, V> RegularizationPathSolver<Problem<&M, D, P>> for ProximalGradient<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    D: SmoothDatafit + NullModelDatafit,
    P: PathPenalty,
    V: ProximalVector,
{
    type Fit = ProximalFit;
    type Penalty = P::AtStrength;
    type Error = FitError<M::Error>;
    fn maximum_penalty(&self, problem: &Problem<&M, D, P>) -> Result<PenaltyMaximum, Self::Error> {
        self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new(prepared);
        let (_, gradient) = null_state(&mut design, problem.datafit())?;
        problem
            .penalty()
            .maximum_penalty(&gradient)
            .map_err(|e| FitError::from_component(e, 0))
    }
    fn solve_regularization_path(
        &self,
        problem: &Problem<&M, D, P>,
        options: &PathOptions,
    ) -> Result<PathFit<ProximalFit, P::AtStrength>, Self::Error> {
        options.validate()?;
        let threshold = self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let design = NativeDesign::<_, V>::new(prepared);
        automatic(
            design,
            problem.datafit(),
            problem.penalty(),
            self,
            options,
            threshold,
        )
    }
}

impl<D, P, Q> PathSolver<Problem<Box<dyn RuntimeDesign + '_>, D, P>, Q>
    for ProximalGradient<Vec<f64>>
where
    D: SmoothDatafit,
    Q: ProximalPenalty,
{
    type Fit = ProximalFit;
    type Error = FitError<BackendError>;
    fn solve_path(
        &self,
        problem: &Problem<Box<dyn RuntimeDesign + '_>, D, P>,
        penalties: Vec<Q>,
    ) -> Result<PathFit<ProximalFit, Q>, Self::Error> {
        let threshold = self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        if penalties.is_empty() {
            return Ok(PathFit::explicit(Vec::new(), penalties));
        }
        let mut design = problem.design().prepare(
            problem.normalization(),
            problem.centers(),
            problem.scales(),
            problem.datafit().fits_intercept(),
        )?;
        let solutions = iterate_path(
            design.as_mut(),
            problem.datafit(),
            &penalties,
            self,
            threshold,
        )?;
        let preprocessing = design.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|solution| ProximalFit::from_solution(solution, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::explicit(fits, penalties))
    }
}

impl<D, P> RegularizationPathSolver<Problem<Box<dyn RuntimeDesign + '_>, D, P>>
    for ProximalGradient<Vec<f64>>
where
    D: SmoothDatafit + NullModelDatafit,
    P: PathPenalty,
{
    type Fit = ProximalFit;
    type Penalty = P::AtStrength;
    type Error = FitError<BackendError>;
    fn maximum_penalty(
        &self,
        problem: &Problem<Box<dyn RuntimeDesign + '_>, D, P>,
    ) -> Result<PenaltyMaximum, Self::Error> {
        self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let mut design = problem.design().prepare(
            problem.normalization(),
            problem.centers(),
            problem.scales(),
            problem.datafit().fits_intercept(),
        )?;
        let (_, gradient) = null_state(design.as_mut(), problem.datafit())?;
        problem
            .penalty()
            .maximum_penalty(&gradient)
            .map_err(|e| FitError::from_component(e, 0))
    }
    fn solve_regularization_path(
        &self,
        problem: &Problem<Box<dyn RuntimeDesign + '_>, D, P>,
        options: &PathOptions,
    ) -> Result<PathFit<ProximalFit, P::AtStrength>, Self::Error> {
        options.validate()?;
        let threshold = self.validate()?;
        validate_rows(problem.design().nrows(), problem.datafit())?;
        let mut design = problem.design().prepare(
            problem.normalization(),
            problem.centers(),
            problem.scales(),
            problem.datafit().fits_intercept(),
        )?;
        let (intercept, gradient) = null_state(design.as_mut(), problem.datafit())?;
        let sequence = options
            .sequence(problem.penalty(), &gradient, design.nrows(), design.ncols())
            .map_err(|e| FitError::from_component(e, 0))?;
        let mut work = Workspace::new(design.nrows(), design.ncols());
        work.intercept = intercept;
        let solutions = sequence
            .penalties
            .iter()
            .map(|penalty| {
                iterate_with_workspace(
                    design.as_mut(),
                    problem.datafit(),
                    penalty,
                    self,
                    threshold,
                    &mut work,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let preprocessing = design.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|solution| ProximalFit::from_solution(solution, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::automatic(fits, sequence))
    }
}

fn validate_rows<D: SmoothDatafit, E>(nrows: usize, datafit: &D) -> Result<(), FitError<E>> {
    if datafit.nobs() != nrows {
        return Err(invalid(
            "datafit observation count must match training rows",
        ));
    }
    Ok(())
}

fn null_state<O, D>(design: &mut O, datafit: &D) -> Result<(f64, Vec<f64>), FitError<O::Error>>
where
    O: ProximalDesign + ?Sized,
    D: SmoothDatafit + NullModelDatafit,
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
    let predictor = vec![intercepts[0]; design.nrows()];
    let mut derivative = vec![0.0; design.nrows()];
    datafit
        .gradient(&predictor, &mut derivative)
        .map_err(|e| FitError::from_component(e, 0))?;
    for &value in &derivative {
        finite(value, "checking the null gradient", 0)?;
    }
    let mut gradient = vec![0.0; design.ncols()];
    design.transpose(&derivative, &mut gradient)?;
    Ok((intercepts[0], gradient))
}

fn automatic<O, D, P, V>(
    mut design: NativeDesign<O, V>,
    datafit: &D,
    penalty: &P,
    solver: &ProximalGradient<V>,
    options: &PathOptions,
    threshold: f64,
) -> Result<PathFit<ProximalFit, P::AtStrength>, FitError<O::Error>>
where
    O: MatVecInto<V> + MatTransposeVecInto<V>,
    D: SmoothDatafit + NullModelDatafit,
    P: PathPenalty,
    V: ProximalVector,
{
    let (intercept, gradient) = null_state(&mut design, datafit)?;
    let sequence = options
        .sequence(penalty, &gradient, design.nrows(), design.ncols())
        .map_err(|e| FitError::from_component(e, 0))?;
    let mut work = Workspace::new(design.nrows(), design.ncols());
    work.intercept = intercept;
    let solutions = sequence
        .penalties
        .iter()
        .map(|penalty| {
            iterate_with_workspace(&mut design, datafit, penalty, solver, threshold, &mut work)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let preprocessing = design.into_preprocessing();
    let fits = solutions
        .into_iter()
        .map(|solution| ProximalFit::from_solution(solution, preprocessing.clone()))
        .collect::<Result<_, _>>()?;
    Ok(PathFit::automatic(fits, sequence))
}
