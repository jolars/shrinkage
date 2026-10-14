//! Two Gaussian responses with an explicit feature penalty axis.

use faer::{Col, Mat};
use shrinkage::{
    AxisPenalty, CoefficientLayout, Gaussian, MultivariateGaussian, PenaltyAxis, Problem,
    ProximalGradient, Ridge, Termination,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(4, 2, |i, j| {
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]][i][j]
    });
    // Contiguous response columns align with the solver's predictor layout.
    let y = [1.0, 5.0, 4.0, 2.0, -6.0, 2.0, -3.0, -1.0];
    let layout = CoefficientLayout::new(2, 2)?;
    let penalty = Ridge::new(0.5);
    let solver = ProximalGradient::<Col<f64>>::new().tolerance(1e-10);
    let fit = Problem::new(
        &x,
        MultivariateGaussian::new(&y, 4, 2),
        AxisPenalty::new(layout, PenaltyAxis::Features, penalty),
    )
    .fit_with(&solver)?;
    assert_eq!(fit.termination(), Termination::Converged);
    for (response, column) in y.chunks_exact(4).enumerate() {
        let independent = Problem::new(&x, Gaussian::new(column), penalty).fit_with(&solver)?;
        assert_eq!(independent.termination(), Termination::Converged);
        for (&joint, &separate) in fit
            .coefficients_for_response(response)
            .unwrap()
            .iter()
            .zip(independent.coefficients())
        {
            assert!((joint - separate).abs() < 1e-8);
        }
        assert!((fit.intercepts()[response] - independent.intercept()).abs() < 1e-8);
        println!(
            "response {response}: coefficients {:?}, intercept {}",
            fit.coefficients_for_response(response).unwrap(),
            fit.intercepts()[response]
        );
    }
    println!("joint objective: {}", fit.objective());
    println!("response-major predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
