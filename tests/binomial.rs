//! Binomial loss stability and proximal fits.

#[path = "common/matrix.rs"]
mod matrix;
#[path = "binomial/reference.rs"]
mod reference;

use matrix::Matrix;
use shrinkage::{
    Binomial, Datafit, Normalization, Problem, ProximalGradient, Ridge, SmoothDatafit, Termination,
};

#[test]
fn extreme_predictors_keep_loss_and_derivatives_finite() {
    let loss = Binomial::new(&[1.0, 0.0, 0.0, 1.0]);
    let eta = [f64::MAX, -f64::MAX, 1000.0, -1000.0];
    assert_eq!(loss.value(&eta).unwrap(), 500.0);
    let mut gradient = [0.0; 4];
    loss.gradient(&eta, &mut gradient).unwrap();
    assert_eq!(gradient, [0.0, 0.0, 0.25, -0.25]);
}

#[test]
fn averaging_extreme_misclassified_losses_does_not_overflow() {
    let datafit = Binomial::new(&[0.0, 1.0, 0.0, 1.0]);
    let predictor = [f64::MAX, -f64::MAX, f64::MAX, -f64::MAX];
    assert_eq!(datafit.value(&predictor).unwrap(), f64::MAX);
    let mut gradient = [f64::NAN; 4];
    datafit.gradient(&predictor, &mut gradient).unwrap();
    assert_eq!(gradient, [0.25, -0.25, 0.25, -0.25]);
}

#[test]
fn intercept_recovers_empirical_log_odds_with_backtracking() {
    let x = Matrix::from_rows(&[&[0.0], &[0.0], &[0.0], &[0.0]]);
    let fit = Problem::new(&x, Binomial::new(&[0.0, 1.0, 1.0, 1.0]), Ridge::new(0.5))
        .normalize(Normalization::None)
        .fit_with(
            &ProximalGradient::<Vec<f64>>::new()
                .initial_step(100.0)
                .tolerance(1e-10),
        )
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert!((fit.intercept() - 3.0_f64.ln()).abs() < 1e-8);
    assert_eq!(fit.coefficients(), &[0.0]);
}

#[test]
fn gradients_match_finite_differences_and_retain_small_tails() {
    let loss = Binomial::new(&[0.0, 0.3, 1.0]);
    let eta = [-4.0, 0.2, 8.0];
    let mut gradient = [0.0; 3];
    loss.gradient(&eta, &mut gradient).unwrap();
    for j in 0..3 {
        let mut plus = eta;
        let mut minus = eta;
        plus[j] += 1e-5;
        minus[j] -= 1e-5;
        let difference = (loss.value(&plus).unwrap() - loss.value(&minus).unwrap()) / 2e-5;
        assert!((difference - gradient[j]).abs() < 1e-10);
    }
    Binomial::new(&[1.0])
        .gradient(&[40.0], &mut gradient[..1])
        .unwrap();
    assert!(gradient[0] < 0.0);
}

#[test]
fn predictor_derivatives_match_finite_differences_across_both_tails() {
    for target in [0.0, 0.3, 1.0] {
        for eta in [
            -700.0, -100.0, -40.0, -8.0, -0.2, 0.0, 0.2, 8.0, 40.0, 100.0, 700.0,
        ] {
            // A single row prevents another row's loss from hiding a tiny tail.
            let response = [target];
            let loss = Binomial::new(&response);
            let mut gradient = [f64::NAN];
            loss.gradient(&[eta], &mut gradient).unwrap();
            let h = 1e-4;
            let difference =
                (loss.value(&[eta + h]).unwrap() - loss.value(&[eta - h]).unwrap()) / (2.0 * h);
            assert!(
                (difference - gradient[0]).abs() <= 3e-8 * gradient[0].abs(),
                "target={target}, eta={eta}: {difference} != {}",
                gradient[0]
            );
            if (target == 1.0 && eta > 0.0) || (target == 0.0 && eta < 0.0) {
                assert_ne!(gradient[0], 0.0, "representable tails must survive");
            }
        }
    }
}

#[test]
fn coefficient_and_intercept_derivatives_match_finite_differences() {
    let rows = [
        [-40.0, 1.0],
        [-2.0, -0.5],
        [0.0, 2.0],
        [1.0, -1.0],
        [40.0, 0.0],
    ];
    let response = [0.0, 0.3, 1.0, 0.0, 1.0];
    let datafit = Binomial::new(&response);
    let predictor = |parameters: [f64; 3]| {
        rows.map(|row| row[0] * parameters[0] + row[1] * parameters[1] + parameters[2])
    };
    let parameters = [1.5, -0.4, 0.2];
    let mut gradient = [0.0; 5];
    datafit
        .gradient(&predictor(parameters), &mut gradient)
        .unwrap();
    for j in 0..3 {
        let derivative: f64 = rows
            .iter()
            .zip(gradient)
            .map(|(row, value)| value * if j == 2 { 1.0 } else { row[j] })
            .sum();
        let mut plus = parameters;
        let mut minus = parameters;
        plus[j] += 1e-5;
        minus[j] -= 1e-5;
        let difference = (datafit.value(&predictor(plus)).unwrap()
            - datafit.value(&predictor(minus)).unwrap())
            / 2e-5;
        assert!((derivative - difference).abs() < 1e-10, "parameter {j}");
    }
}

#[test]
fn invalid_data_and_dimensions_are_rejected() {
    for response in [
        vec![],
        vec![-0.1],
        vec![1.1],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        let loss = Binomial::new(&response);
        assert!(loss.value(&vec![0.0; response.len()]).is_err());
        assert!(
            loss.gradient(&vec![0.0; response.len()], &mut vec![0.0; response.len()])
                .is_err()
        );
    }
    let loss = Binomial::new(&[0.0]);
    assert!(loss.value(&[]).is_err());
    assert!(loss.value(&[f64::INFINITY]).is_err());
    assert!(loss.gradient(&[0.0], &mut []).is_err());
    assert_eq!(loss.response(), &[0.0]);
    assert!(loss.fits_intercept());
    assert!(!loss.fit_intercept(false).fits_intercept());
}

#[test]
fn coefficient_fit_matches_scalar_optimality_equation() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let fit = Problem::new(
        &x,
        Binomial::new(&[0.0, 1.0]).fit_intercept(false),
        Ridge::new(0.2),
    )
    .normalize(Normalization::None)
    .fit_with(
        &ProximalGradient::<Vec<f64>>::new()
            .initial_step(100.0)
            .tolerance(1e-10),
    )
    .unwrap();
    // Symmetry reduces the stationarity equation to 0.2*b = 1/(1+exp(b)).
    let mut lower: f64 = 0.0;
    let mut upper: f64 = 5.0;
    for _ in 0..60 {
        let mid = (lower + upper) / 2.0;
        if 0.2 * mid < 1.0 / (1.0 + mid.exp()) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    assert_eq!(fit.termination(), Termination::Converged);
    assert!((fit.coefficients()[0] - (lower + upper) / 2.0).abs() < 1e-8);
    assert_eq!(fit.intercept(), 0.0);
}

#[test]
fn normalized_and_runtime_fits_preserve_logits() {
    use shrinkage::{MatrixDesign, RuntimeProblem};
    let x = Matrix::from_rows(&[&[8.0], &[8.0], &[12.0], &[12.0]]);
    let normalized = Matrix::from_rows(&[&[-1.0], &[-1.0], &[1.0], &[1.0]]);
    let y = [0.0, 0.0, 0.0, 1.0];
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    let fit = Problem::new(&x, Binomial::new(&y), Ridge::new(0.2))
        .fit_with(&solver)
        .unwrap();
    let explicit = Problem::new(&normalized, Binomial::new(&y), Ridge::new(0.2))
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
    let runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
        Box::new(Binomial::new(&y)),
        Box::new(Ridge::new(0.2)),
    )
    .fit_with(&solver)
    .unwrap();
    for ((a, b), c) in fit
        .predict(&x)
        .unwrap()
        .iter()
        .zip(explicit.predict(&normalized).unwrap())
        .zip(runtime.predict(&x).unwrap())
    {
        assert!((a - b).abs() < 1e-8);
        assert!((a - c).abs() < 1e-8);
    }
    assert!(
        Problem::new(&Matrix::empty(0, 1), Binomial::new(&[]), Ridge::new(0.2))
            .fit_with(&solver)
            .is_err()
    );
}
