//! Independent scikit-learn fixtures for the public Gaussian lasso API.

use super::matrix::Matrix;
use shrinkage::{Lasso, Normalization, Termination};

const X: [[f64; 4]; 8] = [
    [0.0, 1.0, 3.0, 0.0],
    [1.0, 0.0, 3.0, 2.0],
    [2.0, 2.0, 3.0, 0.0],
    [3.0, -1.0, 3.0, 1.0],
    [4.0, 3.0, 3.0, 0.0],
    [5.0, 1.0, 3.0, 2.0],
    [6.0, 4.0, 3.0, 1.0],
    [7.0, -2.0, 3.0, 0.0],
];
const Y: [f64; 8] = [1.0, 2.3, 0.8, 3.9, 4.2, 2.7, 6.1, 5.4];
const PREDICT_X: [[f64; 4]; 2] = [[1.5, -0.5, 3.0, 1.0], [8.0, 2.0, 3.0, 0.0]];

fn close(case: &str, field: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-7 * (1.0 + expected.abs()),
        "{case}: {field}: {actual} != {expected}"
    );
}

#[test]
fn fits_match_independent_sklearn_fixtures() {
    let x = Matrix::from_rows(&X.iter().map(|row| row.as_slice()).collect::<Vec<_>>());
    let predict_x = Matrix::from_rows(
        &PREDICT_X
            .iter()
            .map(|row| row.as_slice())
            .collect::<Vec<_>>(),
    );
    let mut lines = include_str!("../fixtures/lasso_sklearn.csv")
        .lines()
        .filter(|line| !line.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some(
            "case,normalization,fit_intercept,alpha,intercept,coefficient_0,coefficient_1,coefficient_2,coefficient_3,objective,prediction_0,prediction_1"
        )
    );
    let mut cases = 0;
    for line in lines {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 12, "malformed fixture: {line}");
        let case = fields[0];
        let normalization = match fields[1] {
            "none" => Normalization::None,
            "auto" => Normalization::Auto,
            other => panic!("unknown normalization: {other}"),
        };
        let fit_intercept = match fields[2] {
            "0" => false,
            "1" => true,
            other => panic!("unknown intercept policy: {other}"),
        };
        let values: Vec<f64> = fields[3..]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        let fit = Lasso::new(values[0])
            .normalize(normalization)
            .fit_intercept(fit_intercept)
            .tolerance(1e-10)
            .max_iterations(100_000)
            .fit(&x, &Y)
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged, "{case}");
        close(case, "intercept", fit.intercept(), values[1]);
        for (j, (&actual, &expected)) in fit.coefficients().iter().zip(&values[2..6]).enumerate() {
            close(case, &format!("coefficient_{j}"), actual, expected);
        }
        close(case, "objective", fit.objective(), values[6]);
        for (j, (&actual, &expected)) in fit
            .predict(&predict_x)
            .unwrap()
            .iter()
            .zip(&values[7..9])
            .enumerate()
        {
            close(case, &format!("prediction_{j}"), actual, expected);
        }
        cases += 1;
    }
    assert_eq!(cases, 6);
}
