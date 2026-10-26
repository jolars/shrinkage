//! Weighted adjacent fusion with a free intercept and normalized columns.

use faer::{Col, Mat};
use shrinkage::{ChainFusedLasso, Gaussian, Hybrid, Problem, ProximalGradient, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(16, 5, |i, j| ((i * 7 + j * 3 + i * j) % 17) as f64 - 8.0);
    let y: Vec<_> = (0..16)
        .map(|i| 2.0 + x[(i, 0)] + x[(i, 1)] - 0.5 * x[(i, 3)] - 0.5 * x[(i, 4)])
        .collect();
    let problem = Problem::new(
        &x,
        Gaussian::new(&y),
        ChainFusedLasso::new(0.05, 0.2).with_weights(vec![1.0, 0.5, 0.0, 2.0]),
    );
    let fit = problem.fit_with(&Hybrid::<Col<f64>>::new().tolerance(1e-8))?;
    let reference = problem.fit_with(&ProximalGradient::<Col<f64>>::new().tolerance(1e-8))?;
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(reference.termination(), Termination::Converged);
    assert!((fit.objective() - reference.objective()).abs() < 1e-8);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Intercept: {}", fit.intercept());
    println!("Segment labels: {:?}", fit.cluster_labels());
    println!(
        "Optimization-scale segment values: {:?}",
        fit.cluster_values()
    );
    println!(
        "PG steps: {}, segment sweeps: {}",
        fit.pg_steps(),
        fit.coordinate_sweeps()
    );
    println!(
        "Objective: {}, full mapping: {}",
        fit.objective(),
        fit.stopping_value()
    );
    Ok(())
}
