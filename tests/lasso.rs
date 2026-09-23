//! Analytical cases and the backend-independent public fitting contract.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{Lasso, LassoError, Normalization, Termination};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-8 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn default_fit_soft_thresholds_optimization_coefficients() {
    let x = Matrix::from_rows(&[&[0.0], &[2.0], &[4.0]]);
    let y = [1.0, 5.0, 9.0];
    let fit = Lasso::new(0.5).tolerance(1e-10).fit(&x, &y).unwrap();
    let sd = (8.0_f64 / 3.0).sqrt();
    let beta = 2.0 - 0.5 / sd;
    close(fit.coefficients()[0], beta);
    close(fit.intercept(), 5.0 - 2.0 * beta);
    assert_eq!(fit.preprocessing().centers(), Some([2.0].as_slice()));
    close(fit.preprocessing().scales().unwrap()[0], sd);
    assert_eq!(fit.termination(), Termination::Converged);
    assert!(fit.kkt_violation() <= 1e-10);
    close(fit.objective(), 0.125 + 0.5 * beta * sd);
    let new_x = Matrix::from_rows(&[&[-2.0], &[8.0]]);
    let prediction = fit.predict(&new_x).unwrap();
    close(prediction[0], fit.intercept() - 2.0 * beta);
    close(prediction[1], fit.intercept() + 8.0 * beta);
}

#[test]
fn raw_fit_has_unpenalized_intercept_and_averaged_loss() {
    let rows: &[&[f64]] = &[&[0.0], &[2.0], &[4.0]];
    let y = [1.0, 5.0, 9.0];
    let model = Lasso::new(0.5)
        .normalize(Normalization::None)
        .tolerance(1e-10);
    let fit = model.fit(&Matrix::from_rows(rows), &y).unwrap();
    close(fit.coefficients()[0], 2.0 - 0.5 / (8.0 / 3.0));
    close(fit.intercept(), 5.0 - 2.0 * fit.coefficients()[0]);
    assert_eq!(fit.preprocessing().centers(), None);
    assert_eq!(fit.preprocessing().scales(), None);
    let repeated_x = Matrix::from_rows(&rows.repeat(4));
    let repeated = model.fit(&repeated_x, &y.repeat(4)).unwrap();
    close(repeated.coefficients()[0], fit.coefficients()[0]);
    close(repeated.intercept(), fit.intercept());
    close(repeated.objective(), fit.objective());
}

#[test]
fn automatic_normalization_disables_centering_without_intercept() {
    let x = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0]]);
    for normalization in [Normalization::None, Normalization::Auto] {
        let fit = Lasso::new(0.4)
            .fit_intercept(false)
            .normalize(normalization)
            .tolerance(1e-10)
            .fit(&x, &[2.0, 4.0, 6.0])
            .unwrap();
        let scale = if normalization == Normalization::Auto {
            (2.0_f64 / 3.0).sqrt()
        } else {
            1.0
        };
        close(fit.coefficients()[0], 2.0 - 0.4 * scale / (14.0 / 3.0));
        assert_eq!(fit.intercept(), 0.0);
        assert_eq!(fit.preprocessing().centers(), None);
    }
}

#[test]
fn correlated_case_matches_independently_solved_active_set() {
    // The centered Gram matrix is [[1, 1/2], [1/2, 1/2]]. Solving the
    // two active KKT equations at lambda = 1/4 gives beta = [1, 3/2].
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 0.0], &[1.0, 0.0], &[1.0, 1.0]]);
    let y = [0.0, 2.0, 4.0, 6.0];
    let fit = Lasso::new(0.25)
        .normalize(Normalization::None)
        .tolerance(1e-11)
        .fit(&x, &y)
        .unwrap();
    close(fit.coefficients()[0], 1.0);
    close(fit.coefficients()[1], 1.5);
    close(fit.intercept(), 3.0);
    close(fit.objective(), 0.6875);
}

#[test]
fn zero_penalty_and_large_penalty_have_expected_limits() {
    let x = Matrix::from_rows(&[&[0.0], &[2.0], &[4.0]]);
    let y = [1.0, 5.0, 9.0];
    let ols = Lasso::new(0.0).fit(&x, &y).unwrap();
    close(ols.coefficients()[0], 2.0);
    close(ols.intercept(), 1.0);
    close(ols.objective(), 0.0);
    let null = Lasso::new(100.0).fit(&x, &y).unwrap();
    assert_eq!(null.coefficients(), &[0.0]);
    close(null.intercept(), 5.0);
    assert_eq!(null.iterations(), 0);
}

#[test]
fn constant_columns_and_empty_feature_sets_are_supported() {
    let x = Matrix::from_rows(&[&[5.0, 0.0], &[5.0, 0.0], &[5.0, 0.0]]);
    let y = [2.0, 3.0, 4.0];
    let fit = Lasso::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(fit.coefficients(), &[0.0, 0.0]);
    assert_eq!(fit.preprocessing().scales(), Some([1.0, 1.0].as_slice()));
    close(fit.intercept(), 3.0);
    let without_intercept = Lasso::new(0.0).fit_intercept(false).fit(&x, &y).unwrap();
    close(without_intercept.coefficients()[0], 0.6);
    for intercept in [false, true] {
        let fit = Lasso::new(1.0)
            .fit_intercept(intercept)
            .fit(&Matrix::empty(3, 0), &y)
            .unwrap();
        assert!(fit.coefficients().is_empty());
        assert_eq!(fit.intercept(), if intercept { 3.0 } else { 0.0 });
        assert_eq!(fit.termination(), Termination::Converged);
    }
}

#[test]
fn explicit_normalization_and_lazy_fit_preserve_predictions() {
    let x = Matrix::from_rows(&[&[0.0, 2.0], &[2.0, 1.0], &[4.0, 5.0], &[6.0, 0.0]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let lazy = Lasso::new(0.15).tolerance(1e-10).fit(&x, &y).unwrap();
    let centers = lazy.preprocessing().centers().unwrap();
    let scales = lazy.preprocessing().scales().unwrap();
    let rows = [[0.0, 2.0], [2.0, 1.0], [4.0, 5.0], [6.0, 0.0]];
    let normalized: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, value)| (value - centers[j]) / scales[j])
                .collect()
        })
        .collect();
    let explicit_x = Matrix::from_rows(&normalized.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let explicit = Lasso::new(0.15)
        .normalize(Normalization::None)
        .tolerance(1e-10)
        .fit(&explicit_x, &y)
        .unwrap();
    for (a, b) in lazy
        .predict(&x)
        .unwrap()
        .iter()
        .zip(explicit.predict(&explicit_x).unwrap())
    {
        close(*a, b);
    }
    close(lazy.objective(), explicit.objective());
}

#[test]
fn iteration_limit_returns_a_finite_fit_with_diagnostics() {
    let x = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0]]);
    let fit = Lasso::new(0.1)
        .normalize(Normalization::None)
        .tolerance(1e-14)
        .max_iterations(1)
        .fit(&x, &[2.0, 4.0, 6.0])
        .unwrap();
    assert_eq!(fit.termination(), Termination::IterationLimit);
    assert_eq!(fit.iterations(), 1);
    assert!(fit.kkt_violation() > 1e-14);
    assert!(fit.objective().is_finite());
    assert!(fit.predict(&x).unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn invalid_inputs_are_errors() {
    let x = Matrix::from_rows(&[&[1.0], &[2.0]]);
    let y = [1.0, 2.0];
    for lambda in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            Lasso::new(lambda).fit(&x, &y),
            Err(LassoError::InvalidInput { .. })
        ));
    }
    for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Lasso::new(1.0).tolerance(tol).fit(&x, &y).is_err());
    }
    assert!(Lasso::new(1.0).max_iterations(0).fit(&x, &y).is_err());
    assert!(Lasso::new(1.0).fit(&x, &[1.0]).is_err());
    assert!(Lasso::new(1.0).fit(&Matrix::empty(0, 1), &[]).is_err());
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Lasso::new(1.0).fit(&x, &[value, 2.0]).is_err());
        assert!(
            Lasso::new(1.0)
                .fit(&Matrix::from_rows(&[&[value], &[2.0]]), &y)
                .is_err()
        );
    }
    let fit = Lasso::new(0.1).fit(&x, &y).unwrap();
    assert!(fit.predict(&Matrix::empty(2, 2)).is_err());
    assert!(fit.predict(&Matrix::from_rows(&[&[f64::NAN]])).is_err());
    assert!(fit.predict(&Matrix::empty(0, 1)).unwrap().is_empty());
}

#[test]
fn finite_inputs_that_overflow_report_numerical_failure() {
    let x = Matrix::from_rows(&[&[f64::MAX], &[0.0]]);
    assert!(matches!(
        Lasso::new(1.0).fit(&x, &[1.0, 2.0]),
        Err(LassoError::NumericalFailure { .. })
    ));
}

#[test]
fn overflow_during_iteration_or_prediction_is_not_a_successful_fit() {
    let tiny = Matrix::from_rows(&[&[1e-150], &[0.0]]);
    assert!(matches!(
        Lasso::new(0.0)
            .normalize(Normalization::None)
            .fit_intercept(false)
            .fit(&tiny, &[1e160, 0.0]),
        Err(LassoError::NumericalFailure { iteration: 1, .. })
    ));
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let fit = Lasso::new(0.0).fit(&x, &[-2.0, 2.0]).unwrap();
    assert!(matches!(
        fit.predict(&Matrix::from_rows(&[&[f64::MAX]])),
        Err(LassoError::NumericalFailure { .. })
    ));
}

#[test]
fn slow_raw_fit_crosses_residual_refresh_boundaries() {
    let x = Matrix::from_rows(&[&[10.0], &[11.0], &[12.0]]);
    let fit = Lasso::new(0.2)
        .normalize(Normalization::None)
        .tolerance(1e-9)
        .fit(&x, &[21.0, 23.0, 25.0])
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert!(fit.iterations() > 50);
    close(fit.coefficients()[0], 1.7);
    close(fit.intercept(), 4.3);
    close(fit.objective(), 0.37);
}

#[test]
fn negative_coefficients_and_rank_deficient_designs_converge() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[0.0, 0.0], &[1.0, 1.0]]);
    let fit = Lasso::new(0.2)
        .normalize(Normalization::None)
        .tolerance(1e-10)
        .fit(&x, &[3.0, 1.0, -1.0])
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.coefficients().iter().sum(), -1.7);
    close(fit.intercept(), 1.0);
    close(fit.objective(), 0.37);
}
