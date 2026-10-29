//! Generate installed-Python parity fixtures from typed public Rust fits.

use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Gaussian, Hybrid, HybridFit, HybridStep,
    Normalization, Problem, ProximalFit, ProximalGradient, ScreeningDiagnostics, Slope,
    Termination, WorkingSetPolicy,
};

#[derive(Clone)]
struct Case {
    family: &'static str,
    penalty: usize,
    solver: &'static str,
    normalization: &'static str,
    intercept: bool,
    screening: bool,
    policy: WorkingSetPolicy,
    step: HybridStep,
    iterations: usize,
    outer_iterations: usize,
    inner_iterations: usize,
    cluster_sweeps: usize,
    check_every: usize,
    tolerance: f64,
    initial_step: f64,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            family: "gaussian",
            penalty: 0,
            solver: "hybrid",
            normalization: "auto",
            intercept: true,
            screening: true,
            policy: WorkingSetPolicy::Auto,
            step: HybridStep::default(),
            iterations: 10_000,
            outer_iterations: 100,
            inner_iterations: 1_000,
            cluster_sweeps: 5,
            check_every: 10,
            tolerance: 1e-9,
            initial_step: 1.0,
        }
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
    let x = Array2::from_shape_vec((8, 4), rows.iter().flatten().copied().collect())?;
    let predict_x =
        Array2::from_shape_vec((2, 4), predict_rows.iter().flatten().copied().collect())?;
    let mut configurations = Vec::new();
    for family in ["gaussian", "binomial"] {
        for penalty in 0..3 {
            for solver in ["hybrid", "proximal_gradient"] {
                configurations.push(Case {
                    family,
                    penalty,
                    solver,
                    ..Case::default()
                });
                configurations.push(Case {
                    family,
                    penalty,
                    solver,
                    iterations: 1,
                    tolerance: 1e-15,
                    ..Case::default()
                });
            }
        }
        for (index, normalization) in [
            "auto",
            "none",
            "center",
            "standardize",
            "min_max",
            "max_abs",
            "l1",
            "l2",
        ]
        .into_iter()
        .enumerate()
        {
            for intercept in [false, true] {
                for solver in ["hybrid", "proximal_gradient"] {
                    configurations.push(Case {
                        family,
                        penalty: index % 3,
                        solver,
                        normalization,
                        intercept,
                        ..Case::default()
                    });
                }
            }
        }
        for (index, policy) in [
            WorkingSetPolicy::Disabled,
            WorkingSetPolicy::EverActive,
            WorkingSetPolicy::Adaptive {
                initial_size: 1,
                prune: false,
            },
            WorkingSetPolicy::Adaptive {
                initial_size: 2,
                prune: true,
            },
        ]
        .into_iter()
        .enumerate()
        {
            for screening in [false, true] {
                for step in [
                    HybridStep::FixedBound,
                    HybridStep::Backtracking {
                        initial_curvature: Some(0.1),
                        max_backtracks: 100,
                    },
                ] {
                    configurations.push(Case {
                        family,
                        penalty: index % 3,
                        screening,
                        policy,
                        step,
                        cluster_sweeps: 2,
                        check_every: 3,
                        ..Case::default()
                    });
                }
            }
        }
        configurations.push(Case {
            family,
            solver: "proximal_gradient",
            screening: false,
            initial_step: 0.25,
            ..Case::default()
        });
    }
    for (outer_iterations, inner_iterations) in [(1, 1_000), (100, 1)] {
        configurations.push(Case {
            family: "binomial",
            outer_iterations,
            inner_iterations,
            ..Case::default()
        });
    }
    let mut cases = Vec::new();
    for case in configurations {
        let y = if case.family == "gaussian" {
            &gaussian
        } else {
            &binomial
        };
        cases.push(case.evaluate(&x, y, &predict_x)?);
    }
    let fixture = json!({
        "generator": "cargo run --locked --example python_model_fixture --features ndarray_v0_17",
        "x": rows,
        "gaussian_y": gaussian,
        "binomial_y": binomial,
        "predict_x": predict_rows,
        "cases": cases,
    });
    if std::env::args().any(|arg| arg == "--check") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("packages/python/tests/fixtures/model_rust.json");
        let committed: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        assert_close(&fixture, &committed);
    } else {
        println!("{}", serde_json::to_string_pretty(&fixture)?);
    }
    Ok(())
}

impl Case {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y: &[f64],
        predict_x: &Array2<f64>,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let normalization = match self.normalization {
            "auto" => Normalization::Auto,
            "none" => Normalization::None,
            "center" => Normalization::Center,
            "standardize" => Normalization::Standardize,
            "min_max" => Normalization::MinMax,
            "max_abs" => Normalization::MaxAbs,
            "l1" => Normalization::L1,
            "l2" => Normalization::L2,
            _ => unreachable!(),
        };
        let hybrid = Hybrid::<Array1<f64>>::new()
            .tolerance(self.tolerance)
            .max_iterations(self.iterations)
            .max_outer_iterations(self.outer_iterations)
            .max_inner_iterations(self.inner_iterations)
            .cluster_sweeps(self.cluster_sweeps)
            .check_every(self.check_every)
            .screening(self.screening)
            .working_set_policy(self.policy)
            .step_policy(self.step);
        let pg = ProximalGradient::<Array1<f64>>::new()
            .tolerance(self.tolerance)
            .max_iterations(self.iterations)
            .initial_step(self.initial_step)
            .screening(self.screening);
        macro_rules! solve {
            ($datafit:expr, $penalty:expr) => {{
                let problem = Problem::new(x, $datafit, $penalty).normalize(normalization);
                if self.solver == "hybrid" {
                    hybrid_snapshot(&problem.fit_with(&hybrid)?, predict_x)?
                } else {
                    proximal_snapshot(&problem.fit_with(&pg)?, predict_x)?
                }
            }};
        }
        macro_rules! family {
            ($penalty:expr) => {
                if self.family == "gaussian" {
                    solve!(Gaussian::new(y).fit_intercept(self.intercept), $penalty)
                } else {
                    solve!(Binomial::new(y).fit_intercept(self.intercept), $penalty)
                }
            };
        }
        let expected = match self.penalty {
            0 => family!(ClusteredLasso::new(0.03, 0.02)),
            1 => family!(ChainFusedLasso::new(0.03, 0.02).with_weights(vec![0.0, 2.0, 0.5])),
            2 => family!(Slope::new(vec![0.08, 0.04, 0.02, 0.0])),
            _ => unreachable!(),
        };
        let mut options = json!({
            "stopping": {"kind": "proximal_gradient_mapping", "absolute": self.tolerance},
            "max_iterations": self.iterations,
            "screening": self.screening,
        });
        if self.solver == "hybrid" {
            options["max_outer_iterations"] = json!(self.outer_iterations);
            options["max_inner_iterations"] = json!(self.inner_iterations);
            options["cluster_sweeps"] = json!(self.cluster_sweeps);
            options["check_every"] = json!(self.check_every);
            options["working_set_policy"] = policy(self.policy);
            options["step_policy"] = step(self.step);
        } else {
            options["initial_step"] = json!(self.initial_step);
        }
        Ok(json!({
            "family": self.family,
            "penalty": self.penalty,
            "solver": self.solver,
            "normalization": self.normalization,
            "fit_intercept": self.intercept,
            "solver_options": options,
            "expected": expected,
        }))
    }
}

fn policy(policy: WorkingSetPolicy) -> Value {
    match policy {
        WorkingSetPolicy::Auto => json!({"kind": "auto"}),
        WorkingSetPolicy::Disabled => json!({"kind": "disabled"}),
        WorkingSetPolicy::EverActive => json!({"kind": "ever_active"}),
        WorkingSetPolicy::Adaptive {
            initial_size,
            prune,
        } => json!({"kind": "adaptive", "initial_size": initial_size, "prune": prune}),
    }
}

fn step(step: HybridStep) -> Value {
    match step {
        HybridStep::FixedBound => json!({"kind": "fixed_bound"}),
        HybridStep::Backtracking {
            initial_curvature,
            max_backtracks,
        } => json!({
            "kind": "backtracking", "initial_curvature": initial_curvature,
            "max_backtracks": max_backtracks,
        }),
    }
}

fn screening(d: &ScreeningDiagnostics) -> Value {
    json!({
        "supported": d.supported,
        "certificate_passes": d.certificate_passes,
        "zero_tests": d.zero_tests,
        "screened_features": d.screened_features,
        "active_dimension": d.active_dimension,
        "rebuilds": d.rebuilds,
        "fallback_certificates": d.fallback_certificates,
    })
}

macro_rules! common_snapshot {
    ($fit:expr, $x:expr) => {{
        let fit = $fit;
        json!({
            "coefficients": fit.coefficients(),
            "intercept": fit.intercept(),
            "centers": fit.preprocessing().centers(),
            "scales": fit.preprocessing().scales(),
            "termination": match fit.termination() {
                Termination::Converged => "converged",
                Termination::IterationLimit => "iteration_limit",
            },
            "iterations": fit.iterations(),
            "objective": fit.objective(),
            "stopping_value": fit.stopping_value(),
            "stopping_threshold": fit.stopping_threshold(),
            "step_size": fit.step_size(),
            "dual_certificate": fit.dual_certificate(),
            "dual_objective": fit.dual_objective(),
            "duality_gap": fit.duality_gap(),
            "screening_diagnostics": screening(fit.screening_diagnostics()),
            "predictions": fit.predict($x)?,
        })
    }};
}

fn proximal_snapshot(fit: &ProximalFit, x: &Array2<f64>) -> Result<Value, shrinkage::FitError> {
    Ok(common_snapshot!(fit, x))
}

fn hybrid_snapshot(fit: &HybridFit, x: &Array2<f64>) -> Result<Value, shrinkage::FitError> {
    let mut value = common_snapshot!(fit, x);
    value["cluster_labels"] = json!(fit.cluster_labels());
    value["cluster_values"] = json!(fit.cluster_values());
    value["coefficient_signs"] = json!(fit.coefficient_signs());
    value["pg_steps"] = json!(fit.pg_steps());
    value["coordinate_sweeps"] = json!(fit.coordinate_sweeps());
    value["diagnostic_checks"] = json!(fit.diagnostic_checks());
    value["backtracks"] = json!(fit.backtracks());
    value["outer_iterations"] = json!(fit.outer_iterations());
    value["outer_backtracks"] = json!(fit.outer_backtracks());
    value["step_policy"] = step(fit.step_policy());
    let d = fit.working_set_diagnostics();
    value["working_set_diagnostics"] = json!({
        "policy": policy(d.policy),
        "full_checks": d.full_checks,
        "restricted_checks": d.restricted_checks,
        "rebuilds": d.rebuilds,
        "expansions": d.expansions,
        "reactivations": d.reactivations,
        "active_dimension": d.active_dimension,
        "peak_active_dimension": d.peak_active_dimension,
        "feature_visits": d.feature_visits,
        "block_visits": d.block_visits,
    });
    Ok(value)
}

fn assert_close(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "{a} != {b}");
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_close(a, b);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (key, value) in a {
                assert_close(value, &b[key]);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}
