//! Gaussian Group LASSO with explicit group weights and a free intercept.

use faer::{Col, Mat};
use shrinkage::{
    DisjointGroups, Gaussian, GroupLasso, Normalization, Problem, ProximalGradient, Termination,
    UncoveredPolicy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(8, 3, |i: usize, j: usize| {
        if (i & (j + 1)).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    });
    let y: Vec<_> = (0..8)
        .map(|i| 2.0 + 3.0 * x[(i, 0)] - 4.0 * x[(i, 1)])
        .collect();
    let groups = DisjointGroups::new(
        3,
        vec![vec![0, 1], vec![2]],
        vec![2.0, 1.0],
        UncoveredPolicy::Reject,
    )?;
    let fit = Problem::new(&x, Gaussian::new(&y), GroupLasso::new(0.5, groups))
        .normalize(Normalization::None)
        .fit_with(&ProximalGradient::<Col<f64>>::new().tolerance(1e-10))?;
    assert_eq!(fit.termination(), Termination::Converged);
    for (&actual, expected) in fit.coefficients().iter().zip([2.4, -3.2, 0.0]) {
        assert!((actual - expected).abs() < 1e-9);
    }
    assert!((fit.intercept() - 2.0).abs() < 1e-9);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Intercept: {}", fit.intercept());
    println!("Predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
