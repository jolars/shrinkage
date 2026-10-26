//! Dense Gaussian clustered lasso with native products and a free intercept.

use faer::{Col, Mat};
use shrinkage::{ClusteredLasso, Gaussian, Hybrid, Problem, ProximalGradient, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(8, 3, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
    let y: Vec<_> = (0..8)
        .map(|i| 2.0 + x[(i, 0)] + x[(i, 1)] - 0.5 * x[(i, 2)])
        .collect();
    let problem = Problem::new(&x, Gaussian::new(&y), ClusteredLasso::new(0.05, 0.1));
    let fit = problem.fit_with(&Hybrid::<Col<f64>>::new().tolerance(1e-9))?;
    let baseline = problem.fit_with(&ProximalGradient::<Col<f64>>::new().tolerance(1e-9))?;
    assert_eq!(fit.termination(), Termination::Converged);
    assert!((fit.objective() - baseline.objective()).abs() < 1e-8);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Intercept: {}", fit.intercept());
    println!("Optimization-scale clusters: {:?}", fit.cluster_values());
    println!("Feature labels: {:?}", fit.cluster_labels());
    println!(
        "PG steps: {}; coordinate sweeps: {}",
        fit.pg_steps(),
        fit.coordinate_sweeps()
    );
    println!("Full PG mapping: {}", fit.stopping_value());
    println!("Predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
