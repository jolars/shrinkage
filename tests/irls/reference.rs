//! Independent high-precision Newton and active-set binomial fixtures.

use super::matrix::Matrix;
use shrinkage::{Binomial, ElasticNet, Irls, IrlsFit, Normalization, Problem, Termination};

const X: [[f64; 3]; 8] = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 2.0],
    [2.0, 2.0, 0.0],
    [3.0, -1.0, 1.0],
    [4.0, 3.0, 0.0],
    [5.0, 1.0, 2.0],
    [6.0, 4.0, 1.0],
    [7.0, -2.0, 0.0],
];
const EXTREME_X: [[f64; 3]; 8] = [
    [-40.0, 1.0, 0.0],
    [-20.0, 0.0, 2.0],
    [-1.0, 2.0, 0.0],
    [-0.5, -1.0, 1.0],
    [0.5, 3.0, 0.0],
    [1.0, 1.0, 2.0],
    [20.0, 4.0, 1.0],
    [40.0, -2.0, 0.0],
];
const Y: [f64; 8] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
const EXTREME_Y: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
const FRACTIONAL_Y: [f64; 8] = [0.05, 0.2, 0.1, 0.9, 0.75, 0.35, 0.8, 0.95];
const PREDICT_X: [[f64; 3]; 4] = [
    [1.5, -0.5, 1.0],
    [8.0, 2.0, 0.0],
    [-1000.0, 1.0, 2.0],
    [1000.0, -1.0, 0.0],
];

fn close(case: &str, field: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-7 * (1.0 + expected.abs()),
        "{case}: {field}: {actual} != {expected}"
    );
}

fn check_fit(case: &str, fit: &IrlsFit, x: &Matrix, predict_x: &Matrix, expected: &[f64]) {
    assert_eq!(fit.termination(), Termination::Converged, "{case}");
    close(case, "intercept", fit.intercept(), expected[0]);
    for (&actual, &target) in fit.coefficients().iter().zip(&expected[1..4]) {
        close(case, "coefficient", actual, target);
        if target == 0.0 {
            assert_eq!(actual, 0.0, "{case}: inactive coefficient");
        }
    }
    close(case, "objective", fit.objective(), expected[4]);
    let max_logit = fit
        .predict(x)
        .unwrap()
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
    close(case, "max_training_logit", max_logit, expected[5]);
    for (&actual, &target) in fit.predict(predict_x).unwrap().iter().zip(&expected[6..]) {
        close(case, "prediction", actual, target);
    }
}

#[test]
fn irls_matches_independent_decimal_fixtures() {
    let predict_x = Matrix::from_rows(
        &PREDICT_X
            .iter()
            .map(|row| row.as_slice())
            .collect::<Vec<_>>(),
    );
    let mut lines = include_str!("../fixtures/binomial_decimal.csv")
        .lines()
        .filter(|line| !line.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some(
            "case,data,normalization,fit_intercept,l1,l2,intercept,coefficient_0,coefficient_1,coefficient_2,objective,max_training_logit,prediction_0,prediction_1,prediction_2,prediction_3"
        )
    );
    let solver = Irls::new().tolerance(1e-11).max_iterations(200);
    let mut cases = 0;
    let mut extreme_cases = 0;
    for line in lines {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 16, "malformed fixture: {line}");
        let case = fields[0];
        let (rows, response) = match fields[1] {
            "regular" => (&X, &Y),
            "fractional" => (&X, &FRACTIONAL_Y),
            "extreme" => {
                extreme_cases += 1;
                (&EXTREME_X, &EXTREME_Y)
            }
            other => panic!("unknown data: {other}"),
        };
        let x = Matrix::from_rows(&rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>());
        let normalization = match fields[2] {
            "none" => Normalization::None,
            "auto" => Normalization::Auto,
            other => panic!("unknown normalization: {other}"),
        };
        let intercept = match fields[3] {
            "0" => false,
            "1" => true,
            other => panic!("unknown intercept policy: {other}"),
        };
        let values: Vec<f64> = fields[4..]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        let penalty = ElasticNet::new(values[0], values[1]);
        let datafit = Binomial::new(response).fit_intercept(intercept);
        let typed = Problem::new(&x, datafit, penalty)
            .normalize(normalization)
            .fit_with(&solver)
            .unwrap();
        check_fit(case, &typed, &x, &predict_x, &values[2..]);
        if fields[1] == "extreme" {
            assert!(
                values[7] > 40.0,
                "fixture must exercise saturated training logits"
            );
            let logits = typed.predict(&x).unwrap();
            assert!(logits.iter().any(|&eta| eta > 40.0));
            assert!(logits.iter().any(|&eta| eta < -40.0));
        }
        cases += 1;
    }
    assert_eq!(cases, 10);
    assert_eq!(extreme_cases, 1);
}
