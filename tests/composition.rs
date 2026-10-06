//! Component contracts and downstream solver extension without a matrix backend.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    CoordinateDescent, Datafit, FitError, Gaussian, L1, LassoFit, Normalization, Penalty, Problem,
    Solver,
};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12 * (1.0 + expected.abs()));
}

#[test]
fn component_values_use_average_loss_and_optimization_scale_penalty() {
    let loss = Gaussian::new(&[1.0, 4.0, -2.0]);
    close(loss.value(&[2.0, 2.0, 0.0]).unwrap(), 1.5);
    close(L1::new(0.25).value(&[-2.0, 0.0, 4.0]).unwrap(), 1.5);
    close(L1::new(0.0).value(&[f64::MAX, f64::MAX]).unwrap(), 0.0);
    assert_eq!(loss.response(), &[1.0, 4.0, -2.0]);
    assert!(loss.fits_intercept());
    assert!(!loss.fit_intercept(false).fits_intercept());
    assert_eq!(L1::new(0.25).lambda(), 0.25);
}

#[test]
fn component_values_validate_inputs_and_distinguish_overflow() {
    for predictor in [vec![], vec![0.0], vec![0.0, f64::NAN]] {
        assert!(matches!(
            Gaussian::new(&[1.0, 2.0]).value(&predictor),
            Err(FitError::InvalidInput { .. })
        ));
    }
    assert!(Gaussian::new(&[]).value(&[]).is_err());
    assert!(Gaussian::new(&[f64::INFINITY]).value(&[0.0]).is_err());
    assert!(matches!(
        Gaussian::new(&[f64::MAX]).value(&[-f64::MAX]),
        Err(FitError::NumericalFailure { .. })
    ));
    for lambda in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            L1::new(lambda).value(&[1.0]),
            Err(FitError::InvalidInput { .. })
        ));
    }
    assert!(L1::new(0.0).value(&[f64::NAN]).is_err());
    assert!(matches!(
        L1::new(f64::MAX).value(&[2.0]),
        Err(FitError::NumericalFailure { .. })
    ));
}

#[test]
fn extracted_values_reconstruct_the_fitted_objective() {
    let x = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    for intercept in [false, true] {
        let datafit = Gaussian::new(&y).fit_intercept(intercept);
        let penalty = L1::new(0.15);
        let problem = Problem::new(&x, datafit, penalty)
            .with_centers(vec![3.0, 1.25])
            .with_scales(vec![2.0, 0.5]);
        let fit = problem
            .fit_with(&CoordinateDescent::new().tolerance(1e-10))
            .unwrap();
        let theta: Vec<_> = fit
            .coefficients()
            .iter()
            .zip(fit.preprocessing().scales().unwrap())
            .map(|(b, s)| b * s)
            .collect();
        close(
            fit.objective(),
            datafit.value(&fit.predict(&x).unwrap()).unwrap() + penalty.value(&theta).unwrap(),
        );
    }
}

// A downstream solver can inspect the problem and choose its own result type.
struct AuditedCoordinateDescent;

impl Solver<Problem<&Matrix, Gaussian<'_>, L1>> for AuditedCoordinateDescent {
    type Fit = (LassoFit, usize);
    type Error = FitError;

    fn solve(
        &self,
        problem: &Problem<&Matrix, Gaussian<'_>, L1>,
    ) -> Result<Self::Fit, Self::Error> {
        assert_eq!(problem.normalization(), Normalization::None);
        assert_eq!(problem.centers(), Some([2.0].as_slice()));
        assert_eq!(problem.scales(), Some([2.0].as_slice()));
        assert_eq!(problem.penalty().lambda(), 0.0);
        assert!(!problem.datafit().fits_intercept());
        let fit = CoordinateDescent::new().solve(problem)?;
        let predictions = fit.predict(*problem.design())?;
        Ok((fit, predictions.len()))
    }
}

#[test]
fn external_solver_uses_typed_components_and_returns_its_own_result() {
    let x = Matrix::from_rows(&[&[0.0], &[2.0], &[4.0]]);
    let y = [-2.0, 0.0, 2.0];
    let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), L1::new(0.0))
        .normalize(Normalization::None)
        .with_centers(vec![2.0])
        .with_scales(vec![2.0]);
    let (fit, count) = problem.fit_with(&AuditedCoordinateDescent).unwrap();
    assert_eq!(count, 3);
    close(fit.coefficients()[0], 1.0);
    close(fit.intercept(), -2.0);
}

#[test]
fn zero_feature_objective_is_the_gaussian_loss() {
    let x = Matrix::empty(3, 0);
    let datafit = Gaussian::new(&[2.0, 3.0, 4.0]);
    let penalty = L1::new(1.0);
    let fit = Problem::new(&x, datafit, penalty)
        .fit_with(&CoordinateDescent::new())
        .unwrap();
    close(
        fit.objective(),
        datafit.value(&fit.predict(&x).unwrap()).unwrap(),
    );
    close(penalty.value(fit.coefficients()).unwrap(), 0.0);
}
