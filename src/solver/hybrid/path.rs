//! Path orchestration over the same Gaussian and weighted logistic kernels.

use super::*;
use crate::{
    Binomial, NullModelDatafit, PathFit, PathOptions, PathPenalty, PathSolver, PenaltyMaximum,
    RegularizationPathSolver, RuntimeDesign,
};

trait HybridPathDatafit: SmoothDatafit + NullModelDatafit + Copy {
    type Outer;
    fn outer(n: usize, p: usize) -> Self::Outer;
    fn initial_intercept(&self) -> Result<f64, FitError> {
        Ok(0.0)
    }
    fn validate_solver<V>(&self, solver: &Hybrid<V>) -> Result<f64, FitError> {
        solver.validate()
    }
    fn path_deviance<P: HybridPenalty>(
        &self,
        work: &Workspace<P>,
        outer: &Self::Outer,
    ) -> Result<Option<f64>, FitError>;
    fn fit<M, V, P>(
        &self,
        design: &mut NativeDesign<M, V>,
        penalty: &P,
        solver: &Hybrid<V>,
        threshold: f64,
        work: (&mut Workspace<P>, &mut Self::Outer),
    ) -> Result<Solution, FitError<M::Error>>
    where
        M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
        V: ProximalVector,
        P: HybridPenalty;
}

impl HybridPathDatafit for Gaussian<'_> {
    type Outer = ();
    fn outer(_: usize, _: usize) {}
    fn initial_intercept(&self) -> Result<f64, FitError> {
        Ok(self.null_intercepts()?[0])
    }
    fn path_deviance<P: HybridPenalty>(
        &self,
        work: &Workspace<P>,
        _: &(),
    ) -> Result<Option<f64>, FitError> {
        self.deviance(&work.product)
    }
    fn fit<M, V, P>(
        &self,
        design: &mut NativeDesign<M, V>,
        penalty: &P,
        solver: &Hybrid<V>,
        threshold: f64,
        work: (&mut Workspace<P>, &mut ()),
    ) -> Result<Solution, FitError<M::Error>>
    where
        M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
        V: ProximalVector,
        P: HybridPenalty,
    {
        iterate_observed(
            design,
            *self,
            penalty,
            solver,
            threshold,
            work.0,
            &mut NoObserver,
        )
    }
}

impl HybridPathDatafit for Binomial<'_> {
    type Outer = irls::Outer;
    fn outer(n: usize, p: usize) -> Self::Outer {
        irls::Outer::new(n, p)
    }
    fn path_deviance<P: HybridPenalty>(
        &self,
        _: &Workspace<P>,
        outer: &Self::Outer,
    ) -> Result<Option<f64>, FitError> {
        self.deviance(&outer.eta)
    }
    fn validate_solver<V>(&self, solver: &Hybrid<V>) -> Result<f64, FitError> {
        let threshold = solver.validate()?;
        if solver.max_outer_iterations == 0 || solver.max_inner_iterations == 0 {
            return Err(invalid(
                "logistic outer and inner iteration limits must be positive",
            ));
        }
        Ok(threshold)
    }
    fn fit<M, V, P>(
        &self,
        design: &mut NativeDesign<M, V>,
        penalty: &P,
        solver: &Hybrid<V>,
        threshold: f64,
        work: (&mut Workspace<P>, &mut Self::Outer),
    ) -> Result<Solution, FitError<M::Error>>
    where
        M: RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
        V: ProximalVector,
        P: HybridPenalty,
    {
        let (inner, outer) = work;
        irls::iterate_with_workspace_observed(
            design,
            *self,
            penalty,
            solver,
            threshold,
            (inner, outer),
            &mut NoObserver,
        )
    }
}

fn validate<D: SmoothDatafit, E>(n: usize, datafit: &D) -> Result<(), FitError<E>> {
    if n == 0 || datafit.nobs() != n {
        return Err(invalid(
            "datafit observation count must match nonempty training rows",
        ));
    }
    datafit
        .value(&vec![0.0; n])
        .map_err(|e| FitError::from_component(e, 0))?;
    Ok(())
}

impl<M, D, P, Q, V> PathSolver<Problem<&M, D, P>, Q> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    D: HybridPathDatafit,
    Q: HybridPenalty,
    V: ProximalVector,
{
    type Fit = HybridFit;
    type Error = FitError<M::Error>;
    fn solve_path(
        &self,
        problem: &Problem<&M, D, P>,
        penalties: Vec<Q>,
    ) -> Result<PathFit<HybridFit, Q>, Self::Error> {
        let datafit = problem.datafit();
        let threshold = datafit
            .validate_solver(self)
            .map_err(|e| FitError::from_component(e, 0))?;
        let (n, p) = (problem.design().nrows(), problem.design().ncols());
        validate(n, datafit)?;
        let zeros = vec![0.0; p];
        for penalty in &penalties {
            penalty
                .validate_input(&zeros)
                .map_err(|e| FitError::from_component(e, 0))?;
        }
        if penalties.is_empty() {
            return Ok(PathFit::explicit(Vec::new(), penalties));
        }
        let prepared = problem.prepare(datafit.fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        let mut inner = Workspace::<Q>::new(n, p);
        inner.path = true;
        inner.intercept = datafit
            .initial_intercept()
            .map_err(|e| FitError::from_component(e, 0))?;
        let mut outer = D::outer(n, p);
        let solutions = penalties
            .iter()
            .map(|penalty| {
                datafit.fit(
                    &mut design,
                    penalty,
                    self,
                    threshold,
                    (&mut inner, &mut outer),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let preprocessing = design.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|s| HybridFit::from_solution(s, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::explicit(fits, penalties))
    }
}

impl<M, D, P, V> RegularizationPathSolver<Problem<&M, D, P>> for Hybrid<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    D: HybridPathDatafit,
    P: PathPenalty,
    P::AtStrength: HybridPenalty,
    V: ProximalVector,
{
    type Fit = HybridFit;
    type Penalty = P::AtStrength;
    type Error = FitError<M::Error>;
    fn maximum_penalty(&self, problem: &Problem<&M, D, P>) -> Result<PenaltyMaximum, Self::Error> {
        problem
            .datafit()
            .validate_solver(self)
            .map_err(|e| FitError::from_component(e, 0))?;
        validate(problem.design().nrows(), problem.datafit())?;
        let prepared = problem.prepare(problem.datafit().fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        let (_, gradient) = crate::path::null_state(&mut design, problem.datafit())?;
        problem
            .penalty()
            .maximum_penalty(&gradient)
            .map_err(|e| FitError::from_component(e, 0))
    }
    fn solve_regularization_path(
        &self,
        problem: &Problem<&M, D, P>,
        options: &PathOptions,
    ) -> Result<PathFit<HybridFit, P::AtStrength>, Self::Error> {
        options.validate()?;
        let datafit = problem.datafit();
        let threshold = datafit
            .validate_solver(self)
            .map_err(|e| FitError::from_component(e, 0))?;
        let (n, p) = (problem.design().nrows(), problem.design().ncols());
        validate(n, datafit)?;
        let prepared = problem.prepare(datafit.fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new_screenable(prepared);
        let (intercept, gradient) = crate::path::null_state(&mut design, datafit)?;
        let mut sequence = options
            .sequence(problem.penalty(), &gradient, n, p)
            .map_err(|e| FitError::from_component(e, 0))?;
        let null = crate::path::null_deviance(datafit, n)?;
        let mut inner = Workspace::<P::AtStrength>::new(n, p);
        inner.path = true;
        inner.intercept = intercept;
        let mut outer = D::outer(n, p);
        let solutions = sequence.fit(options, null, |penalty| {
            let solution = datafit.fit(
                &mut design,
                penalty,
                self,
                threshold,
                (&mut inner, &mut outer),
            )?;
            let deviance = datafit
                .path_deviance(&inner, &outer)
                .map_err(|e| FitError::from_component(e, 0))?;
            let termination = solution.proximal.termination;
            Ok((solution, deviance, termination))
        })?;
        let preprocessing = design.into_preprocessing();
        let fits = solutions
            .into_iter()
            .map(|s| HybridFit::from_solution(s, preprocessing.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PathFit::automatic(fits, sequence))
    }
}

impl<D: SmoothDatafit, P, Q: crate::ProximalPenalty, V>
    PathSolver<Problem<Box<dyn RuntimeDesign + '_>, D, P>, Q> for Hybrid<V>
{
    type Fit = HybridFit;
    type Error = FitError<BackendError>;
    fn solve_path(
        &self,
        _: &Problem<Box<dyn RuntimeDesign + '_>, D, P>,
        _: Vec<Q>,
    ) -> Result<PathFit<HybridFit, Q>, Self::Error> {
        Err(runtime_unsupported())
    }
}

impl<D: crate::PathDatafit, P: PathPenalty, V>
    RegularizationPathSolver<Problem<Box<dyn RuntimeDesign + '_>, D, P>> for Hybrid<V>
{
    type Fit = HybridFit;
    type Penalty = P::AtStrength;
    type Error = FitError<BackendError>;
    fn maximum_penalty(
        &self,
        _: &Problem<Box<dyn RuntimeDesign + '_>, D, P>,
    ) -> Result<PenaltyMaximum, Self::Error> {
        Err(runtime_unsupported())
    }
    fn solve_regularization_path(
        &self,
        _: &Problem<Box<dyn RuntimeDesign + '_>, D, P>,
        _: &PathOptions,
    ) -> Result<PathFit<HybridFit, Self::Penalty>, Self::Error> {
        Err(runtime_unsupported())
    }
}

fn runtime_unsupported<E>() -> FitError<E> {
    unsupported(
        "Hybrid",
        "erased proximal oracles do not expose cluster or segment columns",
        "use a typed Gaussian or Binomial problem with ClusteredLasso, ChainFusedLasso, or Slope, or runtime ProximalGradient",
    )
}
