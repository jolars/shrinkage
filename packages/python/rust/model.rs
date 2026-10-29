//! Coarse dispatch into typed hybrid and runtime proximal fitting.

use ndarray::Array1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Gaussian, Hybrid, HybridFit, HybridStep,
    MatrixDesign, Normalization, Problem, ProximalFit, ProximalGradient, ProximalPenalty,
    RuntimeProblem, Slope, SmoothDatafit, WorkingSetPolicy,
};

use crate::common::{
    self, DetachedError, fit_error, nonnegative, normalization, parse_stopping, positive,
};
use crate::input::{DenseDesign, DenseInputs};

type PenaltyInput = (String, f64, f64, Option<Vec<f64>>);
type PolicyInput = (String, isize, bool);
type StepInput = (String, Option<f64>, isize);
type PolicyRecord = (&'static str, usize, bool);
type StepRecord = (&'static str, Option<f64>, usize);
type ScreeningRecord = (bool, usize, usize, usize, usize, usize, usize);
type WorkingRecord = (
    PolicyRecord,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

#[derive(Clone, Copy)]
enum Family {
    Gaussian,
    Binomial,
}

impl Family {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "gaussian" => Ok(Self::Gaussian),
            "binomial" => Ok(Self::Binomial),
            _ => Err(PyValueError::new_err("family must be gaussian or binomial")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Binomial => "binomial",
        }
    }
}

enum Penalty {
    Clustered(f64, f64),
    Chain(f64, f64, Option<Vec<f64>>),
    Slope(Vec<f64>),
}

impl Penalty {
    fn parse((name, l1, fusion, weights): PenaltyInput) -> PyResult<Self> {
        match name.as_str() {
            "clustered_lasso" if weights.is_none() => Ok(Self::Clustered(l1, fusion)),
            "chain_fused_lasso" => Ok(Self::Chain(l1, fusion, weights)),
            "slope" => weights
                .map(Self::Slope)
                .ok_or_else(|| PyValueError::new_err("Slope requires explicit rank weights")),
            _ => Err(PyValueError::new_err(
                "penalty must be clustered_lasso, chain_fused_lasso, or slope",
            )),
        }
    }

    fn boxed(self) -> Box<dyn ProximalPenalty> {
        match self {
            Self::Clustered(l1, fusion) => Box::new(ClusteredLasso::new(l1, fusion)),
            Self::Chain(l1, fusion, weights) => Box::new(chain(l1, fusion, weights)),
            Self::Slope(weights) => Box::new(Slope::new(weights)),
        }
    }
}

fn chain(l1: f64, fusion: f64, weights: Option<Vec<f64>>) -> ChainFusedLasso {
    let penalty = ChainFusedLasso::new(l1, fusion);
    match weights {
        Some(weights) => penalty.with_weights(weights),
        None => penalty,
    }
}

fn policy((kind, size, prune): PolicyInput) -> PyResult<WorkingSetPolicy> {
    match kind.as_str() {
        "auto" => Ok(WorkingSetPolicy::Auto),
        "disabled" => Ok(WorkingSetPolicy::Disabled),
        "ever_active" => Ok(WorkingSetPolicy::EverActive),
        "adaptive" => Ok(WorkingSetPolicy::Adaptive {
            initial_size: positive("working-set initial_size", size)?,
            prune,
        }),
        _ => Err(PyValueError::new_err(
            "working-set policy must be auto, disabled, ever_active, or adaptive",
        )),
    }
}

fn policy_record(policy: WorkingSetPolicy) -> PolicyRecord {
    match policy {
        WorkingSetPolicy::Auto => ("auto", 10, true),
        WorkingSetPolicy::Disabled => ("disabled", 10, true),
        WorkingSetPolicy::EverActive => ("ever_active", 10, true),
        WorkingSetPolicy::Adaptive {
            initial_size,
            prune,
        } => ("adaptive", initial_size, prune),
    }
}

fn step((kind, initial_curvature, max_backtracks): StepInput) -> PyResult<HybridStep> {
    match kind.as_str() {
        "backtracking" => Ok(HybridStep::Backtracking {
            initial_curvature,
            max_backtracks: nonnegative("max_backtracks", max_backtracks)?,
        }),
        "fixed_bound" if initial_curvature.is_none() && max_backtracks == 100 => {
            Ok(HybridStep::FixedBound)
        }
        "fixed_bound" => Err(PyValueError::new_err(
            "fixed_bound accepts no backtracking overrides; use HybridStep.fixed_bound()",
        )),
        _ => Err(PyValueError::new_err(
            "step policy must be backtracking or fixed_bound",
        )),
    }
}

#[derive(Clone)]
enum Method {
    Hybrid(Hybrid<Array1<f64>>),
    Proximal(ProximalGradient),
}

impl Method {
    fn solve(
        self,
        data: &DenseInputs,
        family: Family,
        fit_intercept: bool,
        normalization: Normalization,
        penalty: Penalty,
    ) -> Result<NativeFit, DetachedError> {
        match self {
            Self::Hybrid(solver) => {
                // Only these six specialized combinations are instantiated for the hybrid.
                macro_rules! solve {
                    ($penalty:expr) => {
                        match family {
                            Family::Gaussian => Problem::new(
                                &data.design.matrix,
                                Gaussian::new(&data.response).fit_intercept(fit_intercept),
                                $penalty,
                            )
                            .normalize(normalization)
                            .fit_with(&solver),
                            Family::Binomial => Problem::new(
                                &data.design.matrix,
                                Binomial::new(&data.response).fit_intercept(fit_intercept),
                                $penalty,
                            )
                            .normalize(normalization)
                            .fit_with(&solver),
                        }
                    };
                }
                let fit = match penalty {
                    Penalty::Clustered(l1, fusion) => solve!(ClusteredLasso::new(l1, fusion)),
                    Penalty::Chain(l1, fusion, weights) => solve!(chain(l1, fusion, weights)),
                    Penalty::Slope(weights) => solve!(Slope::new(weights)),
                };
                fit.map(|fit| NativeFit::Hybrid(Box::new(fit)))
                    .map_err(DetachedError::from)
            }
            Self::Proximal(solver) => {
                let datafit: Box<dyn SmoothDatafit> = match family {
                    Family::Gaussian => {
                        Box::new(Gaussian::new(&data.response).fit_intercept(fit_intercept))
                    }
                    Family::Binomial => {
                        Box::new(Binomial::new(&data.response).fit_intercept(fit_intercept))
                    }
                };
                let problem = RuntimeProblem::new(
                    Box::new(MatrixDesign::<_, Array1<f64>>::new(&data.design.matrix)),
                    datafit,
                    penalty.boxed(),
                )
                .normalize(normalization);
                problem
                    .fit_with(&solver)
                    .map(|fit| NativeFit::Proximal(Box::new(fit)))
                    .map_err(DetachedError::from)
            }
        }
    }
}

#[pyclass(frozen, name = "_Solver", module = "shrinkage._shrinkage")]
pub(crate) struct PySolver {
    method: Method,
}

#[pymethods]
impl PySolver {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn hybrid(
        stopping: Option<(String, f64, f64)>,
        max_iterations: isize,
        max_outer_iterations: isize,
        max_inner_iterations: isize,
        max_outer_backtracks: isize,
        cluster_sweeps: isize,
        check_every: isize,
        screening: bool,
        working_set_policy: PolicyInput,
        step_policy: StepInput,
    ) -> PyResult<Self> {
        let mut solver = Hybrid::new()
            .max_iterations(positive("max_iterations", max_iterations)?)
            .max_outer_iterations(positive("max_outer_iterations", max_outer_iterations)?)
            .max_inner_iterations(positive("max_inner_iterations", max_inner_iterations)?)
            .max_outer_backtracks(nonnegative("max_outer_backtracks", max_outer_backtracks)?)
            .cluster_sweeps(positive("cluster_sweeps", cluster_sweeps)?)
            .check_every(positive("check_every", check_every)?)
            .screening(screening)
            .working_set_policy(policy(working_set_policy)?)
            .step_policy(step(step_policy)?);
        if let Some(stopping) = stopping {
            solver = solver.terminate_on(parse_stopping(stopping)?);
        }
        Ok(Self {
            method: Method::Hybrid(solver),
        })
    }

    #[staticmethod]
    fn proximal_gradient(
        stopping: Option<(String, f64, f64)>,
        max_iterations: isize,
        initial_step: f64,
        max_backtracks: isize,
        screening: bool,
    ) -> PyResult<Self> {
        let mut solver = ProximalGradient::new()
            .max_iterations(positive("max_iterations", max_iterations)?)
            .initial_step(initial_step)
            .max_backtracks(nonnegative("max_backtracks", max_backtracks)?)
            .screening(screening);
        if let Some(stopping) = stopping {
            solver = solver.terminate_on(parse_stopping(stopping)?);
        }
        Ok(Self {
            method: Method::Proximal(solver),
        })
    }

    #[pyo3(signature = (x, y, family, fit_intercept, normalization_name, penalty, *, copy = true))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        family: &str,
        fit_intercept: bool,
        normalization_name: &str,
        penalty: PenaltyInput,
        copy: bool,
    ) -> PyResult<PyModelFit> {
        let family = Family::parse(family)?;
        let normalization = normalization(normalization_name)?;
        let penalty = Penalty::parse(penalty)?;
        let method = self.method.clone();
        let data = DenseInputs::from_python(x, y, copy)?;
        // Penalties and runtime oracles are created only after detaching. Neither
        // their mutable scratch nor borrowed Python storage crosses this boundary.
        let fit = py
            .detach(move || method.solve(&data, family, fit_intercept, normalization, penalty))
            .map_err(DetachedError::into_python)?;
        Ok(PyModelFit { fit, family })
    }
}

enum NativeFit {
    Hybrid(Box<HybridFit>),
    Proximal(Box<ProximalFit>),
}

#[pyclass(frozen, name = "_ModelFit", module = "shrinkage._shrinkage")]
pub(crate) struct PyModelFit {
    fit: NativeFit,
    family: Family,
}

macro_rules! fitted {
    ($this:expr, $method:ident) => {
        match &$this.fit {
            NativeFit::Hybrid(fit) => fit.$method(),
            NativeFit::Proximal(fit) => fit.$method(),
        }
    };
}

impl PyModelFit {
    fn hybrid_fit(&self) -> PyResult<&HybridFit> {
        match &self.fit {
            NativeFit::Hybrid(fit) => Ok(fit),
            NativeFit::Proximal(_) => Err(PyValueError::new_err(
                "hybrid diagnostics are available only for Hybrid fits",
            )),
        }
    }
}

#[pymethods]
impl PyModelFit {
    #[getter]
    fn solver(&self) -> &'static str {
        match self.fit {
            NativeFit::Hybrid(_) => "hybrid",
            NativeFit::Proximal(_) => "proximal_gradient",
        }
    }
    #[getter]
    fn family(&self) -> &'static str {
        self.family.name()
    }
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        fitted!(self, coefficients).to_vec()
    }
    #[getter]
    fn intercept(&self) -> f64 {
        fitted!(self, intercept)
    }
    #[getter]
    fn centers(&self) -> Option<Vec<f64>> {
        fitted!(self, preprocessing).centers().map(<[f64]>::to_vec)
    }
    #[getter]
    fn scales(&self) -> Option<Vec<f64>> {
        fitted!(self, preprocessing).scales().map(<[f64]>::to_vec)
    }
    #[getter]
    fn centering(&self) -> &'static str {
        common::centering(fitted!(self, preprocessing).centering())
    }
    #[getter]
    fn scaling(&self) -> &'static str {
        common::scaling(fitted!(self, preprocessing).scaling())
    }
    #[getter]
    fn termination(&self) -> &'static str {
        common::termination(fitted!(self, termination))
    }
    #[getter]
    fn iterations(&self) -> usize {
        fitted!(self, iterations)
    }
    #[getter]
    fn objective(&self) -> f64 {
        fitted!(self, objective)
    }
    #[getter]
    fn stopping_value(&self) -> f64 {
        fitted!(self, stopping_value)
    }
    #[getter]
    fn stopping_threshold(&self) -> f64 {
        fitted!(self, stopping_threshold)
    }
    #[getter]
    fn step_size(&self) -> f64 {
        fitted!(self, step_size)
    }
    #[getter]
    fn dual_certificate(&self) -> Option<Vec<f64>> {
        fitted!(self, dual_certificate).map(<[f64]>::to_vec)
    }
    #[getter]
    fn dual_objective(&self) -> Option<f64> {
        fitted!(self, dual_objective)
    }
    #[getter]
    fn duality_gap(&self) -> Option<f64> {
        fitted!(self, duality_gap)
    }
    #[getter]
    fn screening_diagnostics(&self) -> ScreeningRecord {
        let d = fitted!(self, screening_diagnostics);
        (
            d.supported,
            d.certificate_passes,
            d.zero_tests,
            d.screened_features,
            d.active_dimension,
            d.rebuilds,
            d.fallback_certificates,
        )
    }
    #[getter]
    fn working_set_diagnostics(&self) -> PyResult<WorkingRecord> {
        let d = self.hybrid_fit()?.working_set_diagnostics();
        Ok((
            policy_record(d.policy),
            d.full_checks,
            d.restricted_checks,
            d.rebuilds,
            d.expansions,
            d.reactivations,
            d.active_dimension,
            d.peak_active_dimension,
            d.feature_visits,
            d.block_visits,
        ))
    }
    #[getter]
    fn cluster_labels(&self) -> PyResult<Vec<usize>> {
        Ok(self.hybrid_fit()?.cluster_labels().to_vec())
    }
    #[getter]
    fn cluster_values(&self) -> PyResult<Vec<f64>> {
        Ok(self.hybrid_fit()?.cluster_values().to_vec())
    }
    #[getter]
    fn coefficient_signs(&self) -> PyResult<Vec<f64>> {
        Ok(self.hybrid_fit()?.coefficient_signs().to_vec())
    }
    #[getter]
    fn pg_steps(&self) -> PyResult<usize> {
        Ok(self.hybrid_fit()?.pg_steps())
    }
    #[getter]
    fn coordinate_sweeps(&self) -> PyResult<usize> {
        Ok(self.hybrid_fit()?.coordinate_sweeps())
    }
    #[getter]
    fn diagnostic_checks(&self) -> PyResult<usize> {
        Ok(self.hybrid_fit()?.diagnostic_checks())
    }
    #[getter]
    fn backtracks(&self) -> PyResult<usize> {
        Ok(self.hybrid_fit()?.backtracks())
    }
    #[getter]
    fn outer_iterations(&self) -> PyResult<usize> {
        Ok(self.hybrid_fit()?.outer_iterations())
    }
    #[getter]
    fn outer_backtracks(&self) -> PyResult<usize> {
        Ok(self.hybrid_fit()?.outer_backtracks())
    }
    #[getter]
    fn step_policy(&self) -> PyResult<StepRecord> {
        Ok(match self.hybrid_fit()?.step_policy() {
            HybridStep::Backtracking {
                initial_curvature,
                max_backtracks,
            } => ("backtracking", initial_curvature, max_backtracks),
            HybridStep::FixedBound => ("fixed_bound", None, 100),
        })
    }

    #[pyo3(signature = (x, *, copy = true))]
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>, copy: bool) -> PyResult<Vec<f64>> {
        let data = DenseDesign::from_python(x, copy)?;
        py.detach(|| match &self.fit {
            NativeFit::Hybrid(fit) => fit.predict(&data.matrix),
            NativeFit::Proximal(fit) => fit.predict(&data.matrix),
        })
        .map_err(fit_error)
    }
}
