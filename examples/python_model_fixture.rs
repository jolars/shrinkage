//! Generate installed-Python parity fixtures from typed public Rust fits.

#[path = "common/python.rs"]
mod python;

use python::{assert_close, hybrid_snapshot, policy, proximal_snapshot, step};

use ndarray::{Array1, Array2, ShapeBuilder};
use serde_json::{Value, json};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Gaussian, Hybrid, HybridStep, Normalization,
    Problem, ProximalGradient, Slope, WorkingSetPolicy,
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
    // Match the binding's column-major storage to use the same reduction order.
    let x = Array2::from_shape_fn((8, 4).f(), |(i, j)| rows[i][j]);
    let predict_x = Array2::from_shape_fn((2, 4).f(), |(i, j)| predict_rows[i][j]);
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
