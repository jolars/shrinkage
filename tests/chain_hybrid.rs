//! Gaussian reference fits and hybrid contracts for weighted adjacent fusion.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use serde_json::Value;
use shrinkage::{
    ChainFusedLasso, FitError, Gaussian, Hybrid, HybridStep, MatrixDesign, Normalization, Penalty,
    Problem, ProximalGradient, ProximalPenalty, RuntimeProblem, StoppingCriterion, Termination,
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn vector(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn mapping(
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    free_intercept: bool,
    penalty: &ChainFusedLasso,
    step: f64,
) -> f64 {
    let residual: Vec<_> = rows
        .iter()
        .zip(y)
        .map(|(row, &y)| row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>() + intercept - y)
        .collect();
    let input: Vec<_> = beta
        .iter()
        .enumerate()
        .map(|(j, b)| {
            b - step
                * rows
                    .iter()
                    .zip(&residual)
                    .map(|(row, r)| row[j] * r / y.len() as f64)
                    .sum::<f64>()
        })
        .collect();
    let mut output = vec![0.0; beta.len()];
    penalty.prox(&input, step, &mut output).unwrap();
    let initial = if free_intercept {
        residual.iter().sum::<f64>().abs() / y.len() as f64
    } else {
        0.0
    };
    beta.iter()
        .zip(output)
        .map(|(b, z)| (b - z).abs() / step)
        .fold(initial, f64::max)
}

#[test]
fn gaussian_pg_and_hybrid_match_independent_qp_fixtures() {
    let fixtures: Value =
        serde_json::from_str(include_str!("chain_fused_lasso/reference.json")).unwrap();
    for case in fixtures["fits"].as_array().unwrap() {
        let rows: Vec<_> = case["design"]
            .as_array()
            .unwrap()
            .iter()
            .map(vector)
            .collect();
        let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let y = vector(&case["response"]);
        let weights = vector(&case["weights"]);
        let l1 = case["l1"].as_f64().unwrap();
        let fusion = case["fusion"].as_f64().unwrap();
        let free_intercept = case["fit_intercept"].as_bool().unwrap();
        let penalty = ChainFusedLasso::new(l1, fusion).with_weights(weights.clone());
        let datafit = Gaussian::new(&y).fit_intercept(free_intercept);
        let problem = Problem::new(&x, datafit, penalty.clone()).normalize(Normalization::None);
        let solver = ProximalGradient::<Vec<f64>>::new()
            .tolerance(1e-9)
            .max_iterations(100_000);
        let fit = problem.fit_with(&solver).unwrap();
        assert_eq!(
            fit.termination(),
            Termination::Converged,
            "{}",
            case["name"]
        );
        close(fit.objective(), case["objective"].as_f64().unwrap(), 1e-8);
        close(
            fit.stopping_value(),
            mapping(
                &rows,
                &y,
                fit.coefficients(),
                fit.intercept(),
                free_intercept,
                &penalty,
                fit.step_size(),
            ),
            1e-12,
        );
        for (&actual, expected) in fit.predict(&x).unwrap().iter().zip(vector(&case["fitted"])) {
            close(actual, expected, 5e-6);
        }
        let runtime = RuntimeProblem::new(
            Box::new(MatrixDesign::new(&x)),
            Box::new(datafit),
            Box::new(penalty.clone()),
        )
        .normalize(Normalization::None);
        let erased = runtime.fit_with(&solver).unwrap();
        close(erased.objective(), fit.objective(), 1e-12);
        for policy in [HybridStep::default(), HybridStep::FixedBound] {
            let hybrid = problem
                .fit_with(
                    &Hybrid::<Vec<f64>>::new()
                        .tolerance(1e-9)
                        .max_iterations(100_000)
                        .step_policy(policy),
                )
                .unwrap();
            assert_eq!(
                hybrid.termination(),
                Termination::Converged,
                "{}: {}",
                case["name"],
                hybrid.stopping_value()
            );
            close(
                hybrid.objective(),
                case["objective"].as_f64().unwrap(),
                1e-8,
            );
            close(
                hybrid.stopping_value(),
                mapping(
                    &rows,
                    &y,
                    hybrid.coefficients(),
                    hybrid.intercept(),
                    free_intercept,
                    &penalty,
                    hybrid.step_size(),
                ),
                2e-12,
            );
            assert!(hybrid.stopping_value() <= hybrid.stopping_threshold());
            assert_eq!(
                hybrid.iterations(),
                hybrid.pg_steps() + hybrid.coordinate_sweeps()
            );
            for (&actual, expected) in hybrid
                .predict(&x)
                .unwrap()
                .iter()
                .zip(vector(&case["fitted"]))
            {
                close(actual, expected, 5e-6);
            }
            for (j, &label) in hybrid.cluster_labels().iter().enumerate() {
                assert_eq!(hybrid.coefficients()[j], hybrid.cluster_values()[label]);
                if j > 0 {
                    let previous = hybrid.cluster_labels()[j - 1];
                    assert!(label == previous || label == previous + 1);
                    if weights[j - 1] == 0.0 {
                        assert_ne!(label, previous);
                    } else {
                        assert_eq!(
                            label == previous,
                            hybrid.coefficients()[j] == hybrid.coefficients()[j - 1]
                        );
                    }
                }
            }
        }

        // Check the oracle's duals against the original weighted edges, without
        // using Shrinkage's penalty or proximal map.
        let beta = vector(&case["coefficients"]);
        let intercept = case["intercept"].as_f64().unwrap();
        let residual: Vec<_> = rows
            .iter()
            .zip(&y)
            .map(|(row, y)| row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>() + intercept - y)
            .collect();
        let l1_dual = vector(&case["l1_dual"]);
        let fusion_dual = vector(&case["fusion_dual"]);
        for j in 0..beta.len() {
            let mut stationarity = rows
                .iter()
                .zip(&residual)
                .map(|(row, r)| row[j] * r / y.len() as f64)
                .sum::<f64>()
                + l1_dual[j];
            assert!(l1_dual[j].abs() <= l1 + 1e-9);
            close(l1_dual[j] * beta[j], l1 * beta[j].abs(), 1e-8);
            if j > 0 {
                stationarity += fusion_dual[j - 1];
            }
            if j < weights.len() {
                stationarity -= fusion_dual[j];
                let difference = beta[j + 1] - beta[j];
                assert!(fusion_dual[j].abs() <= fusion * weights[j] + 1e-9);
                close(
                    fusion_dual[j] * difference,
                    fusion * weights[j] * difference.abs(),
                    1e-8,
                );
            }
            close(stationarity, 0.0, 1e-8);
        }
        if free_intercept {
            close(residual.iter().sum::<f64>() / y.len() as f64, 0.0, 1e-8);
        }
        let independent = residual
            .iter()
            .map(|r| r * r / (2.0 * y.len() as f64))
            .sum::<f64>()
            + l1 * beta.iter().map(|b| b.abs()).sum::<f64>()
            + fusion
                * beta
                    .windows(2)
                    .zip(&weights)
                    .map(|(pair, w)| w * (pair[1] - pair[0]).abs())
                    .sum::<f64>();
        close(independent, case["objective"].as_f64().unwrap(), 1e-12);
        close(
            penalty.value(&beta).unwrap(),
            independent
                - residual
                    .iter()
                    .map(|r| r * r / (2.0 * y.len() as f64))
                    .sum::<f64>(),
            1e-12,
        );
    }
}

#[test]
fn empty_and_single_feature_fits_and_separated_equal_values() {
    let x = Matrix::empty(3, 0);
    let fit = Problem::new(
        &x,
        Gaussian::new(&[1.0, 2.0, 3.0]),
        ChainFusedLasso::new(0.1, 0.2).with_weights(vec![]),
    )
    .fit_with(&Hybrid::<Vec<f64>>::new())
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(fit.intercept(), 2.0);
    assert!(fit.cluster_labels().is_empty());
    assert!(fit.cluster_values().is_empty());
    let x = Matrix::from_rows(&[&[-1.0], &[0.0], &[2.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[-1.0, 1.0, 5.0]),
        ChainFusedLasso::new(0.1, 100.0).with_weights(vec![]),
    );
    let hybrid = problem
        .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
        .unwrap();
    let pg = problem
        .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-10))
        .unwrap();
    close(hybrid.objective(), pg.objective(), 1e-10);
    let x = Matrix::from_rows(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
    for (response, weights, labels) in [
        ([2.0, -2.0, 2.0], vec![1.0, 1.0], vec![0, 1, 2]),
        ([2.0, 2.0, 2.0], vec![0.0, 1.0], vec![0, 1, 1]),
    ] {
        let fit = Problem::new(
            &x,
            Gaussian::new(&response).fit_intercept(false),
            ChainFusedLasso::new(0.1, 0.0).with_weights(weights),
        )
        .normalize(Normalization::None)
        .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
        .unwrap();
        assert_eq!(fit.cluster_labels(), labels);
        if response[1] < 0.0 {
            assert_eq!(fit.cluster_values(), [1.7, -1.7, 1.7]);
        }
    }
}

#[test]
fn lazy_normalization_and_supplied_metadata_preserve_predictions() {
    let rows = [
        vec![1.0, 2.0, -1.0],
        vec![2.0, -1.0, 0.0],
        vec![3.0, 4.0, 1.0],
        vec![-2.0, 1.0, 2.0],
        vec![1.0, 0.0, 0.0],
    ];
    let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let y = [3.0, -1.0, 4.0, 0.0, 2.0];
    let penalty = ChainFusedLasso::new(0.07, 0.03).with_weights(vec![2.0, 0.0]);
    for free_intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Auto,
            Normalization::Center,
            Normalization::Standardize,
        ] {
            for supplied in [false, true] {
                let mut problem = Problem::new(
                    &x,
                    Gaussian::new(&y).fit_intercept(free_intercept),
                    penalty.clone(),
                )
                .normalize(normalization);
                if supplied {
                    problem = problem
                        .with_centers(vec![0.5, -0.25, 0.1])
                        .with_scales(vec![2.0, 3.0, 0.5]);
                }
                let fit = problem
                    .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
                    .unwrap();
                assert_eq!(fit.termination(), Termination::Converged);
                let centers = fit
                    .preprocessing()
                    .centers()
                    .map_or(vec![0.0; 3], <[f64]>::to_vec);
                let scales = fit
                    .preprocessing()
                    .scales()
                    .map_or(vec![1.0; 3], <[f64]>::to_vec);
                let normalized: Vec<Vec<_>> = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .map(|(j, x)| (x - centers[j]) / scales[j])
                            .collect()
                    })
                    .collect();
                let explicit =
                    Matrix::from_rows(&normalized.iter().map(Vec::as_slice).collect::<Vec<_>>());
                let reference_problem = Problem::new(
                    &explicit,
                    Gaussian::new(&y).fit_intercept(free_intercept),
                    penalty.clone(),
                )
                .normalize(Normalization::None);
                let reference = reference_problem
                    .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
                    .unwrap();
                let pg = reference_problem
                    .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-10))
                    .unwrap();
                close(fit.objective(), reference.objective(), 1e-10);
                close(fit.objective(), pg.objective(), 1e-10);
                for (&actual, expected) in fit
                    .predict(&x)
                    .unwrap()
                    .iter()
                    .zip(reference.predict(&explicit).unwrap())
                {
                    close(actual, expected, 2e-8);
                }
                for (j, &label) in fit.cluster_labels().iter().enumerate() {
                    close(
                        fit.coefficients()[j] * scales[j],
                        fit.cluster_values()[label],
                        1e-14,
                    );
                }
                assert_eq!(fit.preprocessing().centers_were_supplied(), supplied);
                assert_eq!(fit.preprocessing().scales_were_supplied(), supplied);
            }
        }
    }
}

#[test]
fn invalid_dimensions_fail_before_preparation_and_runtime_hybrid_remains_unsupported() {
    let x = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);
    for weights in [vec![], vec![f64::NAN], vec![-1.0], vec![1.0, 2.0]] {
        let result = Problem::new(
            &x,
            Gaussian::new(&[1.0, 2.0]),
            ChainFusedLasso::new(0.0, 0.0).with_weights(weights),
        )
        .fit_with(&Hybrid::<Vec<f64>>::new());
        assert!(matches!(result, Err(FitError::InvalidInput { .. })));
        assert_eq!(x.visits.get(), 0);
    }
    let problem = Problem::new(
        &x,
        Gaussian::new(&[1.0, 2.0]),
        ChainFusedLasso::new(0.1, 0.2),
    );
    for options in [
        Hybrid::<Vec<f64>>::new().max_iterations(0),
        Hybrid::new().cluster_sweeps(0),
        Hybrid::new().check_every(0),
        Hybrid::new().tolerance(f64::NAN),
    ] {
        assert!(problem.fit_with(&options).is_err());
    }
    assert!(
        problem
            .fit_with(&Hybrid::<Vec<f64>>::new().terminate_on(StoppingCriterion::duality_gap(1e-6)))
            .is_err()
    );
    let runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::new(&x)),
        Box::new(Gaussian::new(&[1.0, 2.0])),
        Box::new(ChainFusedLasso::new(0.1, 0.2)),
    );
    assert!(matches!(
        runtime.fit_with(&Hybrid::<Vec<f64>>::new()),
        Err(FitError::UnsupportedCombination { .. })
    ));
}
