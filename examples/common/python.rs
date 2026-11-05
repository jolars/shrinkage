//! Shared snapshots for installed-package Rust parity fixtures.

use ndarray::Array2;
use serde_json::{Value, json};
use shrinkage::{
    HybridFit, HybridStep, ProximalFit, ScreeningDiagnostics, Termination, WorkingSetPolicy,
};

pub fn policy(policy: WorkingSetPolicy) -> Value {
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

pub fn step(step: HybridStep) -> Value {
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

pub fn proximal_snapshot(fit: &ProximalFit, x: &Array2<f64>) -> Result<Value, shrinkage::FitError> {
    Ok(common_snapshot!(fit, x))
}

pub fn hybrid_snapshot(fit: &HybridFit, x: &Array2<f64>) -> Result<Value, shrinkage::FitError> {
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

pub fn assert_close(actual: &Value, expected: &Value) {
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
