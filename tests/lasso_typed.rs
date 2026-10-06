//! Original-scale results from the typed Gaussian, L1, and solver composition.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{CoordinateDescent, Gaussian, L1, Lasso, Normalization, Problem, Termination};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn typed_fit_back_transforms_coefficients_and_induced_intercept() {
    let x = Matrix::from_rows(&[&[0.0], &[2.0], &[4.0]]);
    let y = [-2.0, 0.0, 2.0];
    let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), L1::new(0.0))
        .normalize(Normalization::None)
        .with_centers(vec![2.0])
        .with_scales(vec![2.0]);
    let fit = problem.fit_with(&CoordinateDescent::new()).unwrap();

    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.coefficients()[0], 1.0);
    close(fit.intercept(), -2.0);
    assert_eq!(fit.preprocessing().centers(), Some([2.0].as_slice()));
    assert_eq!(fit.preprocessing().scales(), Some([2.0].as_slice()));
    assert!(fit.preprocessing().centers_were_supplied());
    assert!(fit.preprocessing().scales_were_supplied());
    let new_x = Matrix::from_rows(&[&[-2.0], &[6.0]]);
    for (&actual, expected) in fit.predict(&new_x).unwrap().iter().zip([-4.0, 4.0]) {
        close(actual, expected);
    }
}

#[test]
fn typed_and_convenience_fits_agree_with_explicit_centering() {
    let x = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let solver = CoordinateDescent::new()
        .tolerance(1e-10)
        .max_iterations(50_000);
    let new_x = Matrix::from_rows(&[&[-1.0, 2.0], &[8.0, 0.0]]);
    for intercept in [false, true] {
        let problem = Problem::new(
            &x,
            Gaussian::new(&y).fit_intercept(intercept),
            L1::new(0.15),
        )
        .normalize(Normalization::Standardize);
        let typed = problem.fit_with(&solver).unwrap();
        let convenience = Lasso::new(0.15)
            .fit_intercept(intercept)
            .normalize(Normalization::Standardize)
            .tolerance(1e-10)
            .max_iterations(50_000)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(typed.termination(), Termination::Converged);
        for (&actual, &expected) in typed.coefficients().iter().zip(convenience.coefficients()) {
            close(actual, expected);
        }
        close(typed.intercept(), convenience.intercept());
        assert_ne!(typed.intercept(), 0.0);
        for (&actual, &expected) in typed
            .predict(&new_x)
            .unwrap()
            .iter()
            .zip(convenience.predict(&new_x).unwrap().iter())
        {
            close(actual, expected);
        }
    }
}

#[test]
fn typed_lazy_and_explicit_normalization_preserve_new_predictions() {
    let rows: &[&[f64]] = &[&[0.0, 2.0], &[2.0, 1.0], &[4.0, 5.0], &[6.0, 0.0]];
    let x = Matrix::from_rows(rows);
    let y = [1.0, 2.0, -1.0, 4.0];
    let solver = CoordinateDescent::new().tolerance(1e-10);
    let lazy = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), L1::new(0.15))
        .normalize(Normalization::Standardize)
        .fit_with(&solver)
        .unwrap();
    let centers = lazy.preprocessing().centers().unwrap();
    let scales = lazy.preprocessing().scales().unwrap();
    let transform = |row: &[f64]| {
        row.iter()
            .enumerate()
            .map(|(j, value)| (value - centers[j]) / scales[j])
            .collect::<Vec<_>>()
    };
    let normalized_rows = rows.iter().map(|row| transform(row)).collect::<Vec<_>>();
    let normalized_x = Matrix::from_rows(
        &normalized_rows
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>(),
    );
    let explicit = Problem::new(
        &normalized_x,
        Gaussian::new(&y).fit_intercept(false),
        L1::new(0.15),
    )
    .normalize(Normalization::None)
    .fit_with(&solver)
    .unwrap();

    for (j, &coefficient) in lazy.coefficients().iter().enumerate() {
        close(coefficient, explicit.coefficients()[j] / scales[j]);
    }
    close(
        lazy.intercept(),
        -lazy
            .coefficients()
            .iter()
            .zip(centers)
            .map(|(coefficient, center)| coefficient * center)
            .sum::<f64>(),
    );
    let new_rows: &[&[f64]] = &[&[-1.0, 3.0], &[8.0, -2.0]];
    let new_x = Matrix::from_rows(new_rows);
    let normalized_new_rows = new_rows
        .iter()
        .map(|row| transform(row))
        .collect::<Vec<_>>();
    let normalized_new_x = Matrix::from_rows(
        &normalized_new_rows
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>(),
    );
    for (&actual, &expected) in lazy
        .predict(&new_x)
        .unwrap()
        .iter()
        .zip(explicit.predict(&normalized_new_x).unwrap().iter())
    {
        close(actual, expected);
    }
}

#[test]
fn typed_zero_feature_fit_preserves_intercept_policy() {
    let x = Matrix::empty(3, 0);
    let y = [2.0, 3.0, 4.0];
    for (enabled, expected) in [(false, 0.0), (true, 3.0)] {
        let fit = Problem::new(&x, Gaussian::new(&y).fit_intercept(enabled), L1::new(1.0))
            .fit_with(&CoordinateDescent::new())
            .unwrap();
        assert!(fit.coefficients().is_empty());
        close(fit.intercept(), expected);
        assert_eq!(fit.predict(&x).unwrap(), vec![expected; 3]);
    }
}
