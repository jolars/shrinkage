//! Binomial lasso fitting, probability predictions, and automatic paths.

use faer::Mat;
use shrinkage::{Binomial, Irls, L1, PathOptions, Problem, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(8, 2, |i, j| {
        if j == 0 {
            i as f64 - 3.5
        } else {
            (i % 3) as f64
        }
    });
    let y = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(0.05));
    let solver = Irls::new().tolerance(1e-9);
    let fit = problem.fit_with(&solver)?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!(
        "Coefficients: {:?}; intercept: {}",
        fit.coefficients(),
        fit.intercept()
    );
    println!(
        "Outer steps: {}; coordinate sweeps: {}; KKT violation: {}",
        fit.iterations(),
        fit.coordinate_sweeps(),
        fit.kkt_violation()
    );
    let probabilities: Vec<_> = fit
        .predict(&x)?
        .into_iter()
        .map(|logit| {
            let tail = (-logit.abs()).exp();
            if logit >= 0.0 {
                1.0 / (1.0 + tail)
            } else {
                tail / (1.0 + tail)
            }
        })
        .collect();
    println!("Probabilities: {probabilities:?}");
    let path = problem.fit_regularization_path_with(&solver, &PathOptions::new().nlambda(5))?;
    for (&lambda, fit) in path.lambdas().unwrap().iter().zip(path.fits()) {
        assert_eq!(fit.termination(), Termination::Converged);
        println!("lambda={lambda:.5}: {:?}", fit.coefficients());
    }
    Ok(())
}
