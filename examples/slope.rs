//! Equal-magnitude SLOPE clusters with individual signs and a free intercept.

use faer::{Col, Mat};
use shrinkage::{Gaussian, Hybrid, Problem, ProximalGradient, Slope, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(16, 5, |i, j| ((i * 7 + j * 3 + i * j) % 17) as f64 - 8.0);
    let y: Vec<_> = (0..16)
        .map(|i| 2.0 + x[(i, 0)] - x[(i, 1)] + 0.5 * x[(i, 3)] - 0.5 * x[(i, 4)])
        .collect();
    let problem = Problem::new(
        &x,
        Gaussian::new(&y),
        Slope::new(vec![0.5, 0.4, 0.3, 0.2, 0.1]),
    );
    let fit = problem.fit_with(&Hybrid::<Col<f64>>::new().tolerance(1e-8))?;
    let reference = problem.fit_with(&ProximalGradient::<Col<f64>>::new().tolerance(1e-8))?;
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(reference.termination(), Termination::Converged);
    assert!((fit.objective() - reference.objective()).abs() < 1e-8);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Intercept: {}", fit.intercept());
    println!("Magnitude cluster labels: {:?}", fit.cluster_labels());
    println!("Optimization-scale magnitudes: {:?}", fit.cluster_values());
    println!("Coefficient signs: {:?}", fit.coefficient_signs());
    println!(
        "PG steps: {}, cluster sweeps: {}",
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
