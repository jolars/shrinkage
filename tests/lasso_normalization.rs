//! Normalization choices, penalty scales, and prediction-preserving transforms.

#![cfg(feature = "faer_v0_24")]

use faer::Mat;
use shrinkage::{Centering, Lasso, Normalization, Scaling, Termination};

const ROWS: [[f64; 3]; 4] = [
    [-2.0, 5.0, 0.0],
    [0.0, 5.0, 0.0],
    [4.0, 5.0, 0.0],
    [6.0, 5.0, 0.0],
];

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-7 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn verify(normalization: Normalization, center: Centering, scale: Scaling, intercept: bool) {
    let centers = match center {
        Centering::None => None,
        Centering::Mean => Some([2.0, 5.0, 0.0]),
        Centering::Min => Some([-2.0, 5.0, 0.0]),
    };
    let scales = match scale {
        Scaling::None => None,
        Scaling::Sd => Some([10.0_f64.sqrt(), 1.0, 1.0]),
        Scaling::Range => Some([8.0, 1.0, 1.0]),
        Scaling::MaxAbs => Some(match center {
            Centering::None => [6.0, 5.0, 1.0],
            Centering::Mean => [4.0, 1.0, 1.0],
            Centering::Min => [8.0, 1.0, 1.0],
        }),
        Scaling::L1 => Some(match center {
            Centering::None => [12.0, 20.0, 1.0],
            Centering::Mean => [12.0, 1.0, 1.0],
            Centering::Min => [16.0, 1.0, 1.0],
        }),
        Scaling::L2 => Some(match center {
            Centering::None => [56.0_f64.sqrt(), 10.0, 1.0],
            Centering::Mean => [40.0_f64.sqrt(), 1.0, 1.0],
            Centering::Min => [104.0_f64.sqrt(), 1.0, 1.0],
        }),
    };
    let transform = |value: f64, j: usize| {
        (value - centers.map_or(0.0, |c| c[j])) / scales.map_or(1.0, |s| s[j])
    };
    let x = Mat::from_fn(4, 3, |i, j| ROWS[i][j]);
    let explicit_x = Mat::from_fn(4, 3, |i, j| transform(ROWS[i][j], j));
    let y = [-1.0, 2.0, 4.0, 7.0];
    let model = Lasso::new(0.15)
        .fit_intercept(intercept)
        .tolerance(1e-10)
        .max_iterations(50_000);
    let fit = model.clone().normalize(normalization).fit(&x, &y).unwrap();
    let explicit = model
        .normalize(Normalization::None)
        .fit(&explicit_x, &y)
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(explicit.termination(), Termination::Converged);
    assert_eq!(fit.preprocessing().centering(), center);
    assert_eq!(fit.preprocessing().scaling(), scale);
    assert_eq!(
        fit.preprocessing().centers(),
        centers.as_ref().map(|c| c.as_slice())
    );
    assert_eq!(fit.preprocessing().scales().is_some(), scales.is_some());
    if let Some(expected) = scales {
        for (&actual, expected) in fit.preprocessing().scales().unwrap().iter().zip(expected) {
            close(actual, expected);
        }
    }
    for (j, &coefficient) in fit.coefficients().iter().enumerate() {
        close(
            coefficient,
            explicit.coefficients()[j] / scales.map_or(1.0, |s| s[j]),
        );
    }
    close(fit.objective(), explicit.objective());
    let test_rows = [[-4.0, 1.0, 2.0], [8.0, 7.0, -1.0]];
    let new_x = Mat::from_fn(2, 3, |i, j| test_rows[i][j]);
    let explicit_new_x = Mat::from_fn(2, 3, |i, j| transform(test_rows[i][j], j));
    for (actual, expected) in fit
        .predict(&new_x)
        .unwrap()
        .into_iter()
        .zip(explicit.predict(&explicit_new_x).unwrap())
    {
        close(actual, expected);
    }
}

#[test]
fn presets_match_explicit_normalization_and_preserve_predictions() {
    for intercept in [false, true] {
        for (normalization, center, scale) in [
            (
                Normalization::Auto,
                if intercept {
                    Centering::Mean
                } else {
                    Centering::None
                },
                Scaling::Sd,
            ),
            (Normalization::None, Centering::None, Scaling::None),
            (Normalization::Center, Centering::Mean, Scaling::None),
            (Normalization::Standardize, Centering::Mean, Scaling::Sd),
            (Normalization::MinMax, Centering::Min, Scaling::Range),
            (Normalization::MaxAbs, Centering::None, Scaling::MaxAbs),
            (Normalization::L1, Centering::None, Scaling::L1),
            (Normalization::L2, Centering::None, Scaling::L2),
        ] {
            verify(normalization, center, scale, intercept);
        }
    }
}

#[test]
fn custom_centering_and_scaling_match_explicit_fits() {
    for intercept in [false, true] {
        for center in [Centering::None, Centering::Mean, Centering::Min] {
            for scale in [
                Scaling::None,
                Scaling::Sd,
                Scaling::Range,
                Scaling::MaxAbs,
                Scaling::L1,
                Scaling::L2,
            ] {
                verify(
                    Normalization::Custom { center, scale },
                    center,
                    scale,
                    intercept,
                );
            }
        }
    }
}

#[test]
fn explicit_centering_preserves_induced_intercept_regardless_of_builder_order() {
    let x = Mat::from_fn(3, 1, |i, _| (i + 1) as f64);
    let before = Lasso::new(0.0)
        .normalize(Normalization::Center)
        .fit_intercept(false);
    let after = Lasso::new(0.0)
        .fit_intercept(false)
        .normalize(Normalization::Center);
    for model in [before, after] {
        let fit = model.fit(&x, &[2.0, 4.0, 6.0]).unwrap();
        close(fit.coefficients()[0], 2.0);
        close(fit.intercept(), -4.0);
        for (actual, expected) in fit.predict(&x).unwrap().into_iter().zip([-2.0, 0.0, 2.0]) {
            close(actual, expected);
        }
    }
}
