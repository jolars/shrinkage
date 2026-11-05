//! Generate warm-path and failure parity fixtures for installed Python packages.

#[path = "common/python.rs"]
mod python;

use ndarray::{Array1, Array2, ShapeBuilder};
use python::{assert_close, hybrid_snapshot, policy, proximal_snapshot};
use serde_json::{Value, json};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, FitError, Gaussian, HybridCoordinateDescent,
    HybridStep, Normalization, PathFit, PathOptions, PathTermination, PenaltyMaximum, Problem,
    ProximalGradient, Slope, WorkingSetPolicy,
};

struct Case {
    family: &'static str,
    penalty: usize,
    solver: &'static str,
    mode: &'static str,
    screening: bool,
    policy: WorkingSetPolicy,
    normalization: &'static str,
    intercept: bool,
}

trait Record {
    fn record(&self) -> Value;
}

impl Record for ClusteredLasso {
    fn record(&self) -> Value {
        json!({"kind": "clustered_lasso", "l1": self.l1(), "fusion": self.fusion()})
    }
}
impl Record for ChainFusedLasso {
    fn record(&self) -> Value {
        json!({"kind": "chain_fused_lasso", "l1": self.l1(), "fusion": self.fusion(),
            "weights": self.weights()})
    }
}
impl Record for Slope {
    fn record(&self) -> Value {
        json!({"kind": "slope", "weights": self.weights()})
    }
}

fn snapshot<F, P: Record>(
    result: Result<PathFit<F, P>, FitError>,
    x: &Array2<f64>,
    point: impl Fn(&F, &Array2<f64>) -> Result<Value, FitError>,
) -> Result<Value, FitError> {
    let path = match result {
        Ok(path) => path,
        Err(error) => return Ok(error_record(error)),
    };
    let maximum = match path.maximum_penalty() {
        Some(PenaltyMaximum::Unbounded) => json!("unbounded"),
        Some(PenaltyMaximum::Finite(value)) => json!(value),
        None => Value::Null,
    };
    Ok(json!({
        "penalties": path.penalties().iter().map(Record::record).collect::<Vec<_>>(),
        "lambdas": path.lambdas(),
        "maximum_penalty": maximum,
        "lambda_start": path.lambda_start(),
        "termination": match path.termination() {
            PathTermination::Completed => "completed",
            PathTermination::DevianceChange => "deviance_change",
            PathTermination::DevianceRatio => "deviance_ratio",
            _ => unreachable!(),
        },
        "null_deviance": path.null_deviance(),
        "deviances": path.deviances(),
        "fits": path.fits().iter().map(|fit| point(fit, x)).collect::<Result<Vec<_>, _>>()?,
    }))
}

fn error_record(error: FitError) -> Value {
    let kind = match error {
        FitError::InvalidInput { .. } | FitError::UnsupportedCombination { .. } => "ValueError",
        FitError::NumericalFailure { .. } => "FloatingPointError",
        FitError::LineSearchFailure { .. } | FitError::Backend { .. } => "RuntimeError",
    };
    json!({"error": {"kind": kind, "message": error.to_string()}})
}

impl Case {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y: &[f64],
        predict_x: &Array2<f64>,
    ) -> Result<Value, FitError> {
        let automatic = !matches!(
            self.mode,
            "explicit" | "limited" | "invalid_point" | "line_search"
        );
        let limited = matches!(self.mode, "limited" | "limited_auto");
        let limit = if limited { 1 } else { 10_000 };
        let curvature = if self.mode == "line_search" {
            Some(1e-20)
        } else if limited {
            Some(100.0)
        } else {
            None
        };
        let backtracks = if self.mode == "line_search" { 0 } else { 100 };
        let initial_step = if self.mode == "line_search" {
            1e20
        } else if limited {
            0.01
        } else {
            1.0
        };
        let tolerance = if limited { 1e-15 } else { 1e-9 };
        let hybrid = HybridCoordinateDescent::<Array1<f64>>::new()
            .tolerance(tolerance)
            .max_iterations(limit)
            .check_every(1)
            .screening(self.screening)
            .working_set_policy(self.policy)
            .step_policy(HybridStep::Backtracking {
                initial_curvature: curvature,
                max_backtracks: backtracks,
            });
        let pg = ProximalGradient::<Array1<f64>>::new()
            .tolerance(tolerance)
            .max_iterations(limit)
            .initial_step(initial_step)
            .max_backtracks(backtracks)
            .screening(self.screening);
        let mut options_record =
            json!({"nlambda": 4, "lambda_min_ratio": 0.05, "early_stopping": false});
        match self.mode {
            "early_change" => {
                options_record = json!({"nlambda": 8, "lambda_min_ratio": 1.0,
                "lambda_start": 1000.0, "min_points": 2, "deviance_change_tolerance": 0.1})
            }
            "early_ratio" => {
                options_record = json!({"nlambda": 8, "lambda_start": 0.0,
                "min_points": 1, "deviance_ratio_tolerance": 0.2})
            }
            "override" => options_record["lambda_start"] = json!(0.1),
            "limited_auto" => {
                options_record = json!({"nlambda": 6, "lambda_start": 0.1,
                "lambda_min_ratio": 0.01, "min_points": 2,
                "deviance_change_tolerance": 0.1, "deviance_ratio_tolerance": 0.2})
            }
            _ => (),
        }
        let mut options = PathOptions::new();
        if let Some(value) = options_record.get("nlambda") {
            options = options.nlambda(value.as_u64().unwrap() as usize);
        }
        if let Some(value) = options_record.get("lambda_min_ratio") {
            options = options.lambda_min_ratio(value.as_f64().unwrap());
        }
        if let Some(value) = options_record.get("lambda_start") {
            options = options.lambda_start(value.as_f64().unwrap());
        }
        if let Some(value) = options_record.get("min_points") {
            options = options.min_points(value.as_u64().unwrap() as usize);
        }
        if let Some(value) = options_record.get("deviance_change_tolerance") {
            options = options.deviance_change_tolerance(value.as_f64().unwrap());
        }
        if let Some(value) = options_record.get("deviance_ratio_tolerance") {
            options = options.deviance_ratio_tolerance(value.as_f64().unwrap());
        }
        if let Some(value) = options_record.get("early_stopping") {
            options = options.early_stopping(value.as_bool().unwrap());
        }
        let normalization = if self.normalization == "auto" {
            Normalization::Auto
        } else {
            Normalization::None
        };
        let base_record;
        let points_record;
        macro_rules! evaluate {
            ($datafit:expr, $base:expr, $points:expr) => {{
                let base = $base;
                let points = $points;
                base_record = base.record();
                points_record = json!(points.iter().map(Record::record).collect::<Vec<_>>());
                let problem = Problem::new(x, $datafit.fit_intercept(self.intercept), base)
                    .normalize(normalization);
                if self.solver == "hybrid" {
                    let path = if automatic {
                        problem.fit_regularization_path_with(&hybrid, &options)
                    } else {
                        problem.fit_path_with(&hybrid, points)
                    };
                    snapshot(path, predict_x, hybrid_snapshot)?
                } else {
                    let path = if automatic {
                        problem.fit_regularization_path_with(&pg, &options)
                    } else {
                        problem.fit_path_with(&pg, points)
                    };
                    snapshot(path, predict_x, proximal_snapshot)?
                }
            }};
        }
        macro_rules! family {
            ($base:expr, $points:expr) => {
                if self.family == "gaussian" {
                    evaluate!(Gaussian::new(y), $base, $points)
                } else {
                    evaluate!(Binomial::new(y), $base, $points)
                }
            };
        }
        let unbounded = matches!(self.mode, "unbounded" | "override");
        let invalid = self.mode == "invalid_point";
        let expected = match self.penalty {
            0 => family!(
                ClusteredLasso::new(if unbounded { 0.0 } else { 1.0 }, 0.3),
                vec![
                    ClusteredLasso::new(2.0, 1.0),
                    ClusteredLasso::new(if invalid { -0.03 } else { 0.03 }, 0.01),
                    ClusteredLasso::new(0.03, 0.01),
                    ClusteredLasso::new(0.5, 0.03),
                    ClusteredLasso::new(0.0, 0.0)
                ]
            ),
            1 => family!(
                ChainFusedLasso::new(if unbounded { 0.0 } else { 1.0 }, 0.3)
                    .with_weights(vec![0.0, 2.0, 0.5]),
                vec![
                    ChainFusedLasso::new(2.0, 1.0),
                    ChainFusedLasso::new(0.03, 0.01).with_weights(if invalid {
                        vec![0.0]
                    } else {
                        vec![0.0, 2.0, 0.5]
                    }),
                    ChainFusedLasso::new(0.03, 0.01).with_weights(vec![0.0, 2.0, 0.5]),
                    ChainFusedLasso::new(0.5, 0.03).with_weights(vec![1.0, 0.0, 2.0]),
                    ChainFusedLasso::new(0.0, 0.0)
                ]
            ),
            2 => family!(
                Slope::new(if unbounded {
                    vec![0.0; 4]
                } else {
                    vec![1.0, 0.7, 0.3, 0.0]
                }),
                vec![
                    Slope::new(vec![2.0, 1.8, 1.6, 1.4]),
                    Slope::new(if invalid {
                        vec![0.01, 0.03, 0.02, 0.0]
                    } else {
                        vec![0.08, 0.04, 0.02, 0.0]
                    }),
                    Slope::new(vec![0.08, 0.04, 0.02, 0.0]),
                    Slope::new(vec![0.5, 0.1, 0.05, 0.01]),
                    Slope::new(vec![0.0; 4])
                ]
            ),
            _ => unreachable!(),
        };
        let mut solver_options = json!({"max_iterations": limit, "screening": self.screening,
            "stopping": {"kind": "proximal_gradient_mapping", "absolute": tolerance}});
        if self.solver == "hybrid" {
            solver_options["check_every"] = json!(1);
            solver_options["working_set_policy"] = policy(self.policy);
            solver_options["step_policy"] = json!({"kind": "backtracking", "initial_curvature": curvature, "max_backtracks": backtracks});
        } else {
            solver_options["initial_step"] = json!(initial_step);
            solver_options["max_backtracks"] = json!(backtracks);
        }
        Ok(
            json!({"family": self.family, "solver": self.solver, "mode": self.mode,
            "penalty": base_record, "penalties": if automatic {Value::Null} else {points_record},
            "path_options": if automatic {options_record} else {Value::Null},
            "solver_options": solver_options, "normalization": self.normalization,
            "fit_intercept": self.intercept, "expected": expected}),
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = vec![
        vec![0.0, 1.0, 3.0, 0.0],
        vec![1.0, 0.0, 3.0, 2.0],
        vec![2.0, 2.0, 3.0, 0.0],
        vec![3.0, -1.0, 3.0, 1.0],
        vec![4.0, 3.0, 3.0, 0.0],
        vec![5.0, 1.0, 3.0, 2.0],
        vec![6.0, 4.0, 3.0, 1.0],
        vec![7.0, -2.0, 3.0, 0.0],
    ];
    let gaussian = [1.0, 2.3, 0.8, 3.9, 4.2, 2.7, 6.1, 5.4];
    let binomial = [0.1, 0.2, 0.25, 0.3, 0.65, 0.7, 0.8, 0.9];
    let predict_rows = vec![vec![1.5, -0.5, 3.0, 1.0], vec![8.0, 2.0, 3.0, 0.0]];
    let x = Array2::from_shape_fn((8, 4).f(), |(i, j)| rows[i][j]);
    let predict_x = Array2::from_shape_fn((2, 4).f(), |(i, j)| predict_rows[i][j]);
    let mut cases = Vec::new();
    for family in ["gaussian", "binomial"] {
        let y = if family == "gaussian" {
            &gaussian
        } else {
            &binomial
        };
        for penalty in 0..3 {
            for solver in ["hybrid", "proximal_gradient"] {
                for mode in [
                    "explicit",
                    "limited",
                    "limited_auto",
                    "automatic",
                    "early_change",
                    "early_ratio",
                    "unbounded",
                    "override",
                    "invalid_point",
                    "line_search",
                ] {
                    cases.push(
                        Case {
                            family,
                            penalty,
                            solver,
                            mode,
                            screening: true,
                            policy: WorkingSetPolicy::Auto,
                            normalization: "auto",
                            intercept: true,
                        }
                        .evaluate(&x, y, &predict_x)?,
                    );
                }
                for (screening, policy) in [
                    (false, WorkingSetPolicy::Disabled),
                    (true, WorkingSetPolicy::EverActive),
                    (
                        true,
                        WorkingSetPolicy::Adaptive {
                            initial_size: 1,
                            prune: true,
                        },
                    ),
                    (
                        false,
                        WorkingSetPolicy::Adaptive {
                            initial_size: 2,
                            prune: false,
                        },
                    ),
                ] {
                    if solver == "proximal_gradient" && policy != WorkingSetPolicy::Disabled {
                        continue;
                    }
                    cases.push(
                        Case {
                            family,
                            penalty,
                            solver,
                            mode: "explicit",
                            screening,
                            policy,
                            normalization: "none",
                            intercept: false,
                        }
                        .evaluate(&x, y, &predict_x)?,
                    );
                }
                if family == "binomial" {
                    cases.push(
                        Case {
                            family,
                            penalty,
                            solver,
                            mode: "boundary",
                            screening: true,
                            policy: WorkingSetPolicy::Auto,
                            normalization: "auto",
                            intercept: true,
                        }
                        .evaluate(&x, &[1.0; 8], &predict_x)?,
                    );
                }
            }
        }
    }
    let fixture = json!({"generator": "cargo run --locked --example python_path_fixture --features ndarray_v0_17",
        "x": rows, "gaussian_y": gaussian, "binomial_y": binomial, "predict_x": predict_rows, "cases": cases});
    if std::env::args().any(|arg| arg == "--check") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("packages/python/tests/fixtures/path_rust.json");
        let committed: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        assert_close(&fixture, &committed);
    } else {
        println!("{}", serde_json::to_string_pretty(&fixture)?);
    }
    Ok(())
}
