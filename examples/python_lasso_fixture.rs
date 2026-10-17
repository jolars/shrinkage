//! Generate installed-Python parity fixtures from the public Rust API.
//!
//! Run with `--check` to verify the committed fixture, or redirect stdout to
//! `packages/python/tests/fixtures/lasso_rust.json` to regenerate it.

use ndarray::Array2;
use serde_json::{Value, json};
use shrinkage::{Lasso, Normalization, StoppingCriterion, Termination};

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
    let y = [1.0, 2.3, 0.8, 3.9, 4.2, 2.7, 6.1, 5.4];
    let predict_rows = vec![vec![1.5, -0.5, 3.0, 1.0], vec![8.0, 2.0, 3.0, 0.0]];
    let x = Array2::from_shape_vec((8, 4), rows.iter().flatten().copied().collect())?;
    let predict_x =
        Array2::from_shape_vec((2, 4), predict_rows.iter().flatten().copied().collect())?;
    let mut cases = Vec::new();
    for (name, normalization) in [
        ("auto", Normalization::Auto),
        ("none", Normalization::None),
        ("center", Normalization::Center),
        ("standardize", Normalization::Standardize),
        ("min_max", Normalization::MinMax),
        ("max_abs", Normalization::MaxAbs),
        ("l1", Normalization::L1),
        ("l2", Normalization::L2),
    ] {
        for fit_intercept in [true, false] {
            cases.push(case(
                &x,
                &y,
                &predict_x,
                name,
                normalization,
                fit_intercept,
                0.12,
                None,
                10_000,
            )?);
        }
    }
    for (lambda, stopping, max_iterations) in [
        (0.0, None, 10_000),
        (100.0, None, 10_000),
        (0.12, Some(StoppingCriterion::kkt_violation(1e-10)), 10_000),
        (0.12, Some(StoppingCriterion::kkt_violation(1e-15)), 1),
        (
            0.12,
            Some(StoppingCriterion::DualityGap {
                absolute: 1e-12,
                relative: 1e-8,
            }),
            10_000,
        ),
    ] {
        cases.push(case(
            &x,
            &y,
            &predict_x,
            "none",
            Normalization::None,
            true,
            lambda,
            stopping,
            max_iterations,
        )?);
    }
    let fixture = json!({
        "generator": "cargo run --locked --example python_lasso_fixture --features ndarray_v0_17",
        "x": rows,
        "y": y,
        "predict_x": predict_rows,
        "cases": cases,
    });
    if std::env::args().any(|arg| arg == "--check") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("packages/python/tests/fixtures/lasso_rust.json");
        let committed: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        assert_close(&fixture, &committed);
    } else {
        println!("{}", serde_json::to_string_pretty(&fixture)?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn case(
    x: &Array2<f64>,
    y: &[f64],
    predict_x: &Array2<f64>,
    name: &str,
    normalization: Normalization,
    fit_intercept: bool,
    lambda: f64,
    stopping: Option<StoppingCriterion>,
    max_iterations: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    let mut model = Lasso::new(lambda)
        .fit_intercept(fit_intercept)
        .normalize(normalization)
        .max_iterations(max_iterations);
    if let Some(stopping) = stopping {
        model = model.terminate_on(stopping);
    }
    let fit = model.fit(x, y)?;
    Ok(json!({
        "lambda": lambda,
        "normalization": name,
        "fit_intercept": fit_intercept,
        "stopping": stopping.map(criterion),
        "max_iterations": max_iterations,
        "expected": {
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
            "kkt_violation": fit.kkt_violation(),
            "dual_certificate": fit.dual_certificate(),
            "dual_objective": fit.dual_objective(),
            "duality_gap": fit.duality_gap(),
            "reference_loss": fit.reference_loss(),
            "stopping": criterion(fit.stopping_criterion()),
            "stopping_value": fit.stopping_value(),
            "stopping_threshold": fit.stopping_threshold(),
            "predictions": fit.predict(predict_x)?,
        },
    }))
}

fn criterion(stopping: StoppingCriterion) -> Value {
    match stopping {
        StoppingCriterion::DualityGap { absolute, relative } => json!({
            "kind": "duality_gap", "absolute": absolute, "relative": relative,
        }),
        StoppingCriterion::KktViolation { absolute } => json!({
            "kind": "kkt_violation", "absolute": absolute, "relative": 0.0,
        }),
        StoppingCriterion::ProximalGradientMapping { .. } => unreachable!(),
    }
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
