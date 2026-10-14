//! Automatic lasso, elastic-net, and runtime binomial regularization paths.

use faer::{Col, Mat};
use shrinkage::{
    Binomial, ElasticNet, ErasedPathPenalty, Gaussian, L1, Lasso, MatrixDesign, PathOptions,
    Problem, ProximalGradient, RuntimePathProblem, Termination,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(6, 2, |i, j| if j == 0 { i as f64 } else { (i % 2) as f64 });
    let y = [1.0, 3.0, 4.0, 7.0, 8.0, 11.0];
    let options = PathOptions::new().nlambda(5).lambda_min_ratio(0.01);
    let lasso = Lasso::new(0.0).fit_regularization_path(&x, &y, &options)?;
    for ((_, fit), lambda) in lasso.iter().zip(lasso.lambdas().unwrap()) {
        assert_eq!(fit.termination(), Termination::Converged);
        println!(
            "lasso lambda={lambda:.4}, coefficients={:?}",
            fit.coefficients()
        );
    }
    let problem = Problem::new(&x, Gaussian::new(&y), ElasticNet::from_alpha(0.0, 0.5));
    let solver = ProximalGradient::<Col<f64>>::new();
    let net = problem.fit_regularization_path_with(&solver, &options)?;
    for (penalty, fit) in net.iter() {
        assert_eq!(fit.termination(), Termination::Converged);
        println!(
            "elastic net l1={:.4}, l2={:.4}, objective={:.4}",
            penalty.l1(),
            penalty.l2(),
            fit.objective()
        );
    }
    let binary = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let runtime = RuntimePathProblem::new(
        Box::new(MatrixDesign::<_, Col<f64>>::new(&x)),
        Box::new(Binomial::new(&binary)),
        Box::new(ErasedPathPenalty::new(L1::new(0.0))),
    );
    let logistic = runtime
        .fit_regularization_path_with(&ProximalGradient::new().max_iterations(100_000), &options)?;
    for (fit, lambda) in logistic.fits().iter().zip(logistic.lambdas().unwrap()) {
        assert_eq!(fit.termination(), Termination::Converged);
        println!(
            "binomial lambda={lambda:.4}, intercept={:.4}",
            fit.intercept()
        );
    }
    Ok(())
}
