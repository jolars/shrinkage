//! Numerical calculations shared by the browser explorer and native checks.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use shrinkage::{
    Binomial, ClusteredLasso, CoordinateDescent, Datafit, ElasticNet, FitError, Gaussian, Hybrid,
    InitialPoint, Irls, L1, Normalization, ObservedSolver, Penalty, Problem, ProximalGradient,
    ProximalPenalty, Ridge, Slope, SmoothDatafit, SolverObserver, SolverStep, StepKind,
    StoppingCriterion, Termination,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Loss {
    Gaussian,
    Binomial,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Regularizer {
    L1,
    Ridge,
    ElasticNet,
    Slope,
    ClusteredLasso,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Method {
    Coordinate,
    Proximal,
    Irls,
    Hybrid,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub loss: Loss,
    pub penalty: Regularizer,
    pub correlation: f64,
    pub signal: [f64; 2],
    pub strength: f64,
    pub mixing: f64,
    pub rank_ratio: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            loss: Loss::Gaussian,
            penalty: Regularizer::L1,
            correlation: 0.7,
            signal: [2.0, -1.0],
            strength: 0.05,
            mixing: 0.5,
            rank_ratio: 0.5,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if !self.correlation.is_finite()
            || self.correlation.abs() > 0.9
            || self.signal.iter().any(|b| !b.is_finite() || b.abs() > 3.)
            || !self.strength.is_finite()
            || !(0.0..=2.0).contains(&self.strength)
            || !self.mixing.is_finite()
            || !(0.0..=1.0).contains(&self.mixing)
            || !self.rank_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.rank_ratio)
        {
            return Err("Parameters must lie within the explorer's finite control ranges.".into());
        }
        Ok(())
    }

    pub fn supports(&self, method: Method) -> bool {
        match method {
            Method::Proximal => true,
            Method::Coordinate => self.loss == Loss::Gaussian && self.penalty == Regularizer::L1,
            Method::Irls => matches!(
                self.penalty,
                Regularizer::L1 | Regularizer::Ridge | Regularizer::ElasticNet
            ),
            Method::Hybrid => matches!(
                self.penalty,
                Regularizer::Slope | Regularizer::ClusteredLasso
            ),
        }
    }

    pub fn observations(&self) -> (DMatrix<f64>, Vec<f64>) {
        let x = DMatrix::from_fn(32, 2, |i, j| {
            let angle = 2. * std::f64::consts::PI * i as f64 / 32.;
            let u = std::f64::consts::SQRT_2 * angle.cos();
            let v = std::f64::consts::SQRT_2 * angle.sin();
            if j == 0 {
                u
            } else {
                self.correlation * u + (1. - self.correlation.powi(2)).sqrt() * v
            }
        });
        let y = (0..x.nrows())
            .map(|i| {
                let eta = x[(i, 0)] * self.signal[0] + x[(i, 1)] * self.signal[1];
                match self.loss {
                    Loss::Gaussian => eta,
                    Loss::Binomial => 1. / (1. + (-eta).exp()),
                }
            })
            .collect();
        (x, y)
    }
}

macro_rules! with_penalty {
    ($c:expr, $p:ident, $body:expr) => {{
        let c = $c;
        match c.penalty {
            Regularizer::L1 => {
                let $p = L1::new(c.strength);
                $body
            }
            Regularizer::Ridge => {
                let $p = Ridge::new(c.strength);
                $body
            }
            Regularizer::ElasticNet => {
                let $p = ElasticNet::new(c.strength * c.mixing, c.strength * (1. - c.mixing));
                $body
            }
            Regularizer::Slope => {
                let $p = Slope::new(vec![c.strength, c.strength * c.rank_ratio]);
                $body
            }
            Regularizer::ClusteredLasso => {
                let $p = ClusteredLasso::new(c.strength * c.mixing, c.strength * (1. - c.mixing));
                $body
            }
        }
    }};
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Summary {
    pub coefficients: [f64; 2],
    pub objective: f64,
    pub iterations: usize,
    pub converged: bool,
    pub criterion: String,
    pub stopping_value: f64,
    pub stopping_threshold: f64,
}

trait FitSummary {
    fn summary(&self) -> Summary;
}
macro_rules! fit_summary {
    ($($fit:ty),+) => { $(impl FitSummary for $fit {
        fn summary(&self) -> Summary {
            Summary { coefficients: [self.coefficients()[0], self.coefficients()[1]], objective: self.objective(), iterations: self.iterations(), converged: self.termination() == Termination::Converged,
            criterion: match self.stopping_criterion() { StoppingCriterion::DualityGap { .. } => "Duality gap", StoppingCriterion::KktViolation { .. } => "KKT violation", StoppingCriterion::ProximalGradientMapping { .. } => "PG mapping" }.into(),
            stopping_value: self.stopping_value(), stopping_threshold: self.stopping_threshold() }
        }
    })+ };
}
fit_summary!(
    shrinkage::LassoFit,
    shrinkage::ProximalFit,
    shrinkage::IrlsFit,
    shrinkage::HybridFit
);

#[derive(Debug, Serialize, Deserialize)]
pub struct Point {
    pub coefficients: [f64; 2],
    pub objective: f64,
    pub iteration: usize,
    pub kind: String,
    pub inner: bool,
    pub inner_iteration: Option<usize>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Trace {
    pub points: Vec<Point>,
    pub fit: Summary,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ZeroSet {
    Origin,
    Diagonal,
    Plane,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Scene {
    pub radius: f64,
    pub grid_low: f64,
    pub grid_step: f64,
    pub grid_size: usize,
    pub loss: Vec<f64>,
    pub penalty: Vec<f64>,
    pub objective: Vec<f64>,
    pub loss_minimum: f64,
    pub loss_at_optimum: f64,
    pub penalty_budget: f64,
    pub penalty_zero_set: ZeroSet,
    pub reference: Summary,
    pub penalty_clipped: bool,
    pub compatible_methods: Vec<Method>,
}

struct Recorder<F>(F);

impl<F: for<'a> FnMut(SolverStep<'a>, Option<usize>)> SolverObserver for Recorder<F> {
    fn observe(&mut self, step: SolverStep<'_>) {
        self.0(step, None);
    }

    fn observe_inner(&mut self, step: SolverStep<'_>, outer_iteration: usize) {
        self.0(step, Some(outer_iteration));
    }
}

fn run<'a, D, P, S>(
    x: &'a DMatrix<f64>,
    datafit: D,
    penalty: P,
    solver: &S,
    start: [f64; 2],
) -> Result<Trace, String>
where
    D: Datafit<Error = FitError>,
    P: Penalty<Error = FitError>,
    S: ObservedSolver<Problem<&'a DMatrix<f64>, D, P>>,
    S::Fit: FitSummary,
    S::Error: std::fmt::Display,
{
    let problem = Problem::new(x, datafit, penalty).normalize(Normalization::None);
    let mut points = Vec::new();
    let mut eta = vec![0.; x.nrows()];
    let mut observation_error = None;
    let fit = problem
        .fit_from_with_observer(
            solver,
            InitialPoint {
                coefficients: &start,
                intercept: 0.,
            },
            &mut Recorder(|step: SolverStep<'_>, outer: Option<usize>| {
                for (i, value) in eta.iter_mut().enumerate() {
                    *value = step.intercept
                        + x[(i, 0)] * step.coefficients[0]
                        + x[(i, 1)] * step.coefficients[1];
                }
                match problem
                    .datafit()
                    .value(&eta)
                    .and_then(|loss| problem.penalty().value(step.coefficients).map(|p| loss + p))
                {
                    Ok(objective) => points.push(Point {
                        coefficients: [step.coefficients[0], step.coefficients[1]],
                        objective,
                        iteration: outer.unwrap_or(step.iteration),
                        inner: outer.is_some(),
                        inner_iteration: outer.map(|_| step.iteration),
                        kind: match step.kind {
                            StepKind::Initial => "Initial".into(),
                            StepKind::Coordinate { index } => format!("Coordinate {}", index + 1),
                            StepKind::Block { index } => format!("Block {}", index + 1),
                            StepKind::Intercept => "Intercept".into(),
                            StepKind::ProximalGradient => "Proximal step".into(),
                            StepKind::CoordinateSweep => "Cluster sweep".into(),
                            StepKind::Irls => "IRLS step".into(),
                            StepKind::Final(_) => "Final".into(),
                            _ => "Accepted step".into(),
                        },
                    }),
                    Err(error) => observation_error = Some(error.to_string()),
                }
            }),
        )
        .map_err(|e| e.to_string())?;
    if let Some(error) = observation_error {
        return Err(error);
    }
    Ok(Trace {
        points,
        fit: fit.summary(),
    })
}

pub fn trace(config: &Config, method: Method) -> Result<Trace, String> {
    trace_from(config, method, [0., 0.])
}

pub fn trace_from(config: &Config, method: Method, start: [f64; 2]) -> Result<Trace, String> {
    if start.iter().any(|b| !b.is_finite() || b.abs() > 1000.) {
        return Err("Invalid starting point.".into());
    }
    config.validate()?;
    if !config.supports(method) {
        return Err("This datafit, penalty, and solver combination is unsupported.".into());
    }
    let (x, y) = config.observations();
    let gaussian = Gaussian::new(&y).fit_intercept(false);
    let binomial = Binomial::new(&y).fit_intercept(false);
    macro_rules! fit {
        ($p:expr, $s:expr) => {
            match config.loss {
                Loss::Gaussian => run(&x, gaussian, $p, &$s, start),
                Loss::Binomial => run(&x, binomial, $p, &$s, start),
            }
        };
    }
    match method {
        Method::Coordinate => run(
            &x,
            gaussian,
            L1::new(config.strength),
            &CoordinateDescent::new(),
            start,
        ),
        Method::Proximal => {
            with_penalty!(config, p, fit!(p, ProximalGradient::<DVector<f64>>::new()))
        }
        Method::Irls => match config.penalty {
            Regularizer::L1 => fit!(L1::new(config.strength), Irls::new()),
            Regularizer::Ridge => fit!(Ridge::new(config.strength), Irls::new()),
            Regularizer::ElasticNet => fit!(
                ElasticNet::new(
                    config.strength * config.mixing,
                    config.strength * (1. - config.mixing)
                ),
                Irls::new()
            ),
            _ => unreachable!("compatibility checked"),
        },
        Method::Hybrid => match config.penalty {
            Regularizer::Slope => fit!(
                Slope::new(vec![config.strength, config.strength * config.rank_ratio]),
                Hybrid::<DVector<f64>>::new()
            ),
            Regularizer::ClusteredLasso => fit!(
                ClusteredLasso::new(
                    config.strength * config.mixing,
                    config.strength * (1. - config.mixing)
                ),
                Hybrid::<DVector<f64>>::new()
            ),
            _ => unreachable!("compatibility checked"),
        },
    }
}

pub fn scene(config: &Config) -> Result<Scene, String> {
    scene_with_radius(config, 0.)
}

pub fn scene_with_radius(config: &Config, minimum_radius: f64) -> Result<Scene, String> {
    if !minimum_radius.is_finite() || !(0.0..=1000.0).contains(&minimum_radius) {
        return Err("Invalid viewport radius.".into());
    }
    config.validate()?;
    let (x, y) = config.observations();
    with_penalty!(
        config,
        penalty,
        match config.loss {
            Loss::Gaussian => scene_with(
                config,
                minimum_radius,
                &x,
                Gaussian::new(&y).fit_intercept(false),
                penalty
            ),
            Loss::Binomial => scene_with(
                config,
                minimum_radius,
                &x,
                Binomial::new(&y).fit_intercept(false),
                penalty
            ),
        }
    )
}

fn scene_with<D: SmoothDatafit, P: ProximalPenalty>(
    config: &Config,
    minimum_radius: f64,
    x: &DMatrix<f64>,
    datafit: D,
    penalty: P,
) -> Result<Scene, String> {
    let problem = Problem::new(x, datafit, penalty).normalize(Normalization::None);
    let reference = problem
        .fit_with(
            &ProximalGradient::<DVector<f64>>::new()
                .tolerance(1e-9)
                .max_iterations(50_000),
        )
        .map_err(|e| e.to_string())?
        .summary();
    let radius = config
        .signal
        .iter()
        .chain(reference.coefficients.iter())
        .fold(1_f64, |r, b| r.max(b.abs()))
        .ceil()
        .max(minimum_radius.ceil())
        + 1.;
    let grid_size = 163;
    let grid_step = 2. * radius / 161.;
    let grid_low = -radius - grid_step;
    let mut loss = Vec::with_capacity(grid_size * grid_size);
    let mut regularization = Vec::with_capacity(loss.capacity());
    let mut objective = Vec::with_capacity(loss.capacity());
    let mut eta = vec![0.; x.nrows()];
    let mut evaluate = |b: [f64; 2]| -> Result<f64, String> {
        for (i, value) in eta.iter_mut().enumerate() {
            *value = x[(i, 0)] * b[0] + x[(i, 1)] * b[1];
        }
        problem.datafit().value(&eta).map_err(|e| e.to_string())
    };
    let loss_minimum = evaluate(config.signal)?;
    let loss_at_optimum = evaluate(reference.coefficients)?;
    let penalty_budget = problem
        .penalty()
        .value(&reference.coefficients)
        .map_err(|e| e.to_string())?;
    let penalty_zero_set = if config.strength == 0. {
        ZeroSet::Plane
    } else if config.penalty == Regularizer::ClusteredLasso && config.mixing == 0. {
        ZeroSet::Diagonal
    } else {
        ZeroSet::Origin
    };
    let mut penalty_clipped = false;
    for j in 0..grid_size {
        for i in 0..grid_size {
            let b = [
                grid_low + (i as f64 + 0.5) * grid_step,
                grid_low + (j as f64 + 0.5) * grid_step,
            ];
            let l = evaluate(b)?;
            let p = problem.penalty().value(&b).map_err(|e| e.to_string())?;
            if p <= penalty_budget && (i == 1 || j == 1 || i == grid_size - 2 || j == grid_size - 2)
            {
                penalty_clipped = true;
            }
            loss.push(l);
            regularization.push(p);
            objective.push(l + p);
        }
    }
    Ok(Scene {
        radius,
        grid_low,
        grid_step,
        grid_size,
        loss,
        penalty: regularization,
        objective,
        loss_minimum,
        loss_at_optimum,
        penalty_budget,
        penalty_zero_set,
        reference,
        penalty_clipped,
        compatible_methods: [
            Method::Coordinate,
            Method::Proximal,
            Method::Irls,
            Method::Hybrid,
        ]
        .into_iter()
        .filter(|m| config.supports(*m))
        .collect(),
    })
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(js_name = evaluateScene)]
    pub fn evaluate_scene(value: JsValue, minimum_radius: f64) -> Result<JsValue, JsValue> {
        let config: Config =
            serde_wasm_bindgen::from_value(value).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let scene =
            scene_with_radius(&config, minimum_radius).map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&scene).map_err(|e| JsValue::from_str(&e.to_string()))
    }
    #[wasm_bindgen(js_name = fitTrace)]
    pub fn fit_trace(value: JsValue, method: JsValue, start: JsValue) -> Result<JsValue, JsValue> {
        let config: Config =
            serde_wasm_bindgen::from_value(value).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let method: Method = serde_wasm_bindgen::from_value(method)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let start: [f64; 2] =
            serde_wasm_bindgen::from_value(start).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let trace = trace_from(&config, method, start).map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&trace).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
