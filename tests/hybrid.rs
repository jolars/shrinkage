//! Dense Gaussian hybrid contracts and independent epigraph-QP references.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use serde_json::Value;
use shrinkage::{
    ClusteredLasso, FitError, Gaussian, Hybrid, HybridFit, HybridStep, MatrixDesign, Normalization,
    Penalty, Problem, ProximalGradient, ProximalPenalty, RuntimeProblem, StoppingCriterion,
    Termination,
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

fn explicit_objective(
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    l1: f64,
    fusion: f64,
) -> f64 {
    let n = y.len() as f64;
    rows.iter()
        .zip(y)
        .map(|(row, &y)| {
            let r = y - intercept - row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>();
            r * r / (2.0 * n)
        })
        .sum::<f64>()
        + l1 * beta.iter().map(|b| b.abs()).sum::<f64>()
        + fusion
            * (0..beta.len())
                .flat_map(|i| (i + 1..beta.len()).map(move |j| (beta[i] - beta[j]).abs()))
                .sum::<f64>()
}

fn mapping(
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    fit_intercept: bool,
    penalty: &ClusteredLasso,
    step: f64,
) -> f64 {
    let residual: Vec<_> = rows
        .iter()
        .zip(y)
        .map(|(row, &y)| intercept + row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>() - y)
        .collect();
    let input: Vec<_> = beta
        .iter()
        .enumerate()
        .map(|(j, &b)| {
            b - step
                * rows
                    .iter()
                    .zip(&residual)
                    .map(|(row, r)| row[j] * r / y.len() as f64)
                    .sum::<f64>()
        })
        .collect();
    let mut prox = vec![0.0; beta.len()];
    penalty.prox(&input, step, &mut prox).unwrap();
    let initial = if fit_intercept {
        residual.iter().sum::<f64>().abs() / y.len() as f64
    } else {
        0.0
    };
    beta.iter()
        .zip(prox)
        .map(|(b, z)| (b - z).abs() / step)
        .fold(initial, f64::max)
}

#[test]
fn typed_runtime_and_hybrid_fits_match_independent_qp_fixtures() {
    let fixtures: Value =
        serde_json::from_str(include_str!("clustered_lasso/reference.json")).unwrap();
    for case in fixtures["cases"].as_array().unwrap() {
        let rows: Vec<_> = case["design"]
            .as_array()
            .unwrap()
            .iter()
            .map(vector)
            .collect();
        let y = vector(&case["response"]);
        let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let l1 = case["l1"].as_f64().unwrap();
        let fusion = case["fusion"].as_f64().unwrap();
        let free_intercept = case["fit_intercept"].as_bool().unwrap();
        let penalty = ClusteredLasso::new(l1, fusion);
        let datafit = Gaussian::new(&y).fit_intercept(free_intercept);
        let problem = Problem::new(&x, datafit, penalty.clone()).normalize(Normalization::None);
        let expected = vector(&case["fitted"]);
        let objective = case["objective"].as_f64().unwrap();
        let baseline = problem
            .fit_with(
                &ProximalGradient::<Vec<f64>>::new()
                    .tolerance(1e-8)
                    .max_iterations(100_000),
            )
            .unwrap();
        assert_eq!(
            baseline.termination(),
            Termination::Converged,
            "{}: mapping={}, step={}",
            case["name"],
            baseline.stopping_value(),
            baseline.step_size()
        );
        close(baseline.objective(), objective, 5e-9);
        close(
            baseline.stopping_value(),
            mapping(
                &rows,
                &y,
                baseline.coefficients(),
                baseline.intercept(),
                free_intercept,
                &penalty,
                baseline.step_size(),
            ),
            2e-13,
        );
        for (actual, &expected) in baseline.predict(&x).unwrap().iter().zip(&expected) {
            close(*actual, expected, 5e-6);
        }
        let runtime = RuntimeProblem::new(
            Box::new(MatrixDesign::new(&x)),
            Box::new(datafit),
            Box::new(penalty.clone()),
        )
        .normalize(Normalization::None);
        let erased = runtime
            .fit_with(
                &ProximalGradient::new()
                    .tolerance(1e-8)
                    .max_iterations(100_000),
            )
            .unwrap();
        close(erased.objective(), baseline.objective(), 1e-12);
        for policy in [HybridStep::default(), HybridStep::FixedBound] {
            let fit = problem
                .fit_with(
                    &Hybrid::<Vec<f64>>::new()
                        .tolerance(1e-8)
                        .max_iterations(100_000)
                        .step_policy(policy),
                )
                .unwrap();
            assert_eq!(
                fit.termination(),
                Termination::Converged,
                "{}",
                case["name"]
            );
            close(fit.objective(), objective, 5e-9);
            close(
                fit.objective(),
                explicit_objective(&rows, &y, fit.coefficients(), fit.intercept(), l1, fusion),
                1e-13,
            );
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
                2e-13,
            );
            assert!(fit.stopping_value() <= fit.stopping_threshold());
            assert_eq!(fit.iterations(), fit.pg_steps() + fit.coordinate_sweeps());
            for (actual, &expected) in fit.predict(&x).unwrap().iter().zip(&expected) {
                close(*actual, expected, 5e-6);
            }
            for (j, &label) in fit.cluster_labels().iter().enumerate() {
                assert_eq!(fit.coefficients()[j], fit.cluster_values()[label]);
            }
            assert!(
                fit.cluster_values()
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
        }
        // Certify the oracle independently through its explicit-pair duals.
        let beta = vector(&case["coefficients"]);
        let l1_dual = vector(&case["l1_dual"]);
        let fusion_dual = vector(&case["fusion_dual"]);
        let mut subgradient = l1_dual.clone();
        for (&b, &s) in beta.iter().zip(&l1_dual) {
            assert!(s.abs() <= l1 + 1e-8);
            close(s * b, l1 * b.abs(), 1e-8);
        }
        let mut pair = 0;
        for i in 0..beta.len() {
            for j in i + 1..beta.len() {
                let s = fusion_dual[pair];
                pair += 1;
                assert!(s.abs() <= fusion + 1e-8);
                close(
                    s * (beta[i] - beta[j]),
                    fusion * (beta[i] - beta[j]).abs(),
                    1e-8,
                );
                subgradient[i] += s;
                subgradient[j] -= s;
            }
        }
        for j in 0..beta.len() {
            let gradient = rows
                .iter()
                .enumerate()
                .map(|(i, row)| row[j] * (expected[i] - y[i]) / y.len() as f64)
                .sum::<f64>();
            close(gradient + subgradient[j], 0.0, 1e-8);
        }
    }
}

#[test]
fn research_loss_curvature_and_stopping_conversion_preserve_the_maps() {
    let rows = vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![0.5, -1.0]];
    let y = [-2.0, 1.0, 0.5];
    let beta = [0.3, -0.2];
    let (l1, fusion, curvature) = (0.1, 0.2, 7.0);
    let n = y.len() as f64;
    let gradient: Vec<_> = (0..2)
        .map(|j| {
            rows.iter()
                .zip(y)
                .map(|(row, y)| row[j] * (row[0] * beta[0] + row[1] * beta[1] - y))
                .sum::<f64>()
        })
        .collect();
    let input: Vec<_> = beta
        .iter()
        .zip(&gradient)
        .map(|(b, g)| b - g / curvature)
        .collect();
    let mut research = [0.0; 2];
    let mut averaged = [0.0; 2];
    ClusteredLasso::new(l1, fusion)
        .prox(&input, 1.0 / curvature, &mut research)
        .unwrap();
    ClusteredLasso::new(l1 / n, fusion / n)
        .prox(&input, n / curvature, &mut averaged)
        .unwrap();
    for (actual, expected) in averaged.iter().zip(research) {
        close(*actual, expected, 1e-14);
    }
    close(
        explicit_objective(&rows, &y, &beta, 0.0, l1 / n, fusion / n) * n,
        n * explicit_objective(&rows, &y, &beta, 0.0, 0.0, 0.0)
            + ClusteredLasso::new(l1, fusion).value(&beta).unwrap(),
        1e-14,
    );
    let scale = (0..2)
        .map(|j| {
            rows.iter()
                .zip(y)
                .map(|(row, y)| row[j] * y)
                .sum::<f64>()
                .abs()
        })
        .fold(1.0_f64, f64::max);
    let research_mapping = beta
        .iter()
        .zip(research)
        .map(|(b, z)| curvature * (b - z).abs())
        .fold(0.0, f64::max);
    let averaged_mapping = mapping(
        &rows,
        &y,
        &beta,
        0.0,
        false,
        &ClusteredLasso::new(l1 / n, fusion / n),
        n / curvature,
    );
    close(research_mapping, n * averaged_mapping, 1e-14);
    for relative in [1e-8, 0.1, 1.0] {
        assert_eq!(
            research_mapping / scale <= relative,
            averaged_mapping <= relative * scale / n
        );
    }
}

#[test]
fn budgets_cadence_and_backtracking_do_not_change_the_schedule() {
    let x = Matrix::from_rows(&[
        &[1.0, 2.0, 0.0],
        &[0.0, 1.0, 2.0],
        &[2.0, 0.0, 1.0],
        &[1.0, -1.0, 1.0],
    ]);
    let y = [1.0, -2.0, 3.0, 0.7];
    let problem = Problem::new(
        &x,
        Gaussian::new(&y).fit_intercept(false),
        ClusteredLasso::new(0.003, 0.001),
    )
    .normalize(Normalization::None);
    for policy in [
        HybridStep::FixedBound,
        HybridStep::Backtracking {
            initial_curvature: Some(1e-4),
            max_backtracks: 100,
        },
    ] {
        let options = Hybrid::<Vec<f64>>::new()
            .step_policy(policy)
            .tolerance(1e-30)
            .max_iterations(11);
        let first = problem.fit_with(&options.clone().check_every(1)).unwrap();
        let second = problem.fit_with(&options.check_every(100)).unwrap();
        assert_eq!(first.termination(), Termination::IterationLimit);
        assert_eq!(first.iterations(), 11);
        assert_eq!((first.pg_steps(), first.coordinate_sweeps()), (2, 9));
        assert_eq!(first.coefficients(), second.coefficients());
        assert_eq!(first.intercept(), second.intercept());
        assert_eq!(first.step_size(), second.step_size());
        assert_eq!(first.backtracks(), second.backtracks());
        assert_eq!(first.diagnostic_checks(), 12);
        assert_eq!(second.diagnostic_checks(), 2);
    }
    let failed = problem.fit_with(&Hybrid::<Vec<f64>>::new().step_policy(
        HybridStep::Backtracking {
            initial_curvature: Some(1e-4),
            max_backtracks: 0,
        },
    ));
    assert!(matches!(
        failed,
        Err(FitError::LineSearchFailure { iteration: 0, .. })
    ));
}

#[test]
fn zero_and_nonzero_blocks_split_through_full_pg() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let fit = Problem::new(
        &x,
        Gaussian::new(&[-2.0, 2.0]).fit_intercept(false),
        ClusteredLasso::new(0.05, 0.1),
    )
    .normalize(Normalization::None)
    .fit_with(&Hybrid::<Vec<f64>>::new().check_every(1).tolerance(1e-12))
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.coefficients()[0], -1.7, 1e-12);
    close(fit.coefficients()[1], 1.7, 1e-12);
    assert_ne!(fit.cluster_labels()[0], fit.cluster_labels()[1]);
}

fn compare_normalization(fit: &HybridFit, rows: &[Vec<f64>], y: &[f64], free_intercept: bool) {
    let centers = fit
        .preprocessing()
        .centers()
        .map_or(vec![0.0; rows[0].len()], <[f64]>::to_vec);
    let scales = fit
        .preprocessing()
        .scales()
        .map_or(vec![1.0; rows[0].len()], <[f64]>::to_vec);
    let normalized: Vec<Vec<_>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &x)| (x - centers[j]) / scales[j])
                .collect()
        })
        .collect();
    let explicit = Matrix::from_rows(&normalized.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let reference = Problem::new(
        &explicit,
        Gaussian::new(y).fit_intercept(free_intercept),
        ClusteredLasso::new(0.07, 0.03),
    )
    .normalize(Normalization::None)
    .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
    .unwrap();
    close(fit.objective(), reference.objective(), 1e-10);
    let raw = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    for (actual, expected) in fit
        .predict(&raw)
        .unwrap()
        .iter()
        .zip(reference.predict(&explicit).unwrap())
    {
        close(*actual, expected, 1e-8);
    }
    for (j, &label) in fit.cluster_labels().iter().enumerate() {
        close(
            fit.coefficients()[j] * scales[j],
            fit.cluster_values()[label],
            1e-14,
        );
    }
}

#[test]
fn lazy_normalization_and_supplied_metadata_preserve_predictions() {
    let rows = vec![
        vec![1.0, 2.0],
        vec![2.0, -1.0],
        vec![3.0, 4.0],
        vec![-2.0, 1.0],
        vec![1.0, 0.0],
    ];
    let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let y = [3.0, -1.0, 4.0, 0.0, 2.0];
    for free_intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Auto,
            Normalization::Center,
            Normalization::Standardize,
        ] {
            let fit = Problem::new(
                &x,
                Gaussian::new(&y).fit_intercept(free_intercept),
                ClusteredLasso::new(0.07, 0.03),
            )
            .normalize(normalization)
            .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
            .unwrap();
            compare_normalization(&fit, &rows, &y, free_intercept);
        }
        let fit = Problem::new(
            &x,
            Gaussian::new(&y).fit_intercept(free_intercept),
            ClusteredLasso::new(0.07, 0.03),
        )
        .normalize(Normalization::None)
        .with_centers(vec![0.5, -0.25])
        .with_scales(vec![2.0, 3.0])
        .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
        .unwrap();
        compare_normalization(&fit, &rows, &y, free_intercept);
        assert!(fit.preprocessing().centers_were_supplied());
        assert!(fit.preprocessing().scales_were_supplied());
    }
}

#[test]
fn cluster_labels_describe_optimization_scale() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0], &[-1.0, 0.0], &[0.0, -2.0]]);
    let fit = Problem::new(
        &x,
        Gaussian::new(&[2.0, 2.0, -2.0, -2.0]).fit_intercept(false),
        ClusteredLasso::new(0.1, 1.0),
    )
    .normalize(Normalization::None)
    .with_scales(vec![1.0, 2.0])
    .fit_with(&Hybrid::<Vec<f64>>::new())
    .unwrap();
    assert_eq!(fit.cluster_labels()[0], fit.cluster_labels()[1]);
    assert_ne!(fit.coefficients()[0], fit.coefficients()[1]);
}

#[test]
fn empty_features_zero_design_and_invalid_settings() {
    for p in [0, 2] {
        let x = Matrix::empty(3, p);
        for free_intercept in [false, true] {
            let fit = Problem::new(
                &x,
                Gaussian::new(&[1.0, 2.0, 3.0]).fit_intercept(free_intercept),
                ClusteredLasso::new(0.0, 0.0),
            )
            .normalize(Normalization::None)
            .fit_with(&Hybrid::<Vec<f64>>::new())
            .unwrap();
            assert_eq!(fit.termination(), Termination::Converged);
            assert_eq!(fit.iterations(), 0);
            assert_eq!(fit.intercept(), if free_intercept { 2.0 } else { 0.0 });
            assert_eq!(
                fit.predict(&Matrix::empty(0, p)).unwrap(),
                Vec::<f64>::new()
            );
        }
    }
    let x = Matrix::from_rows(&[&[1.0], &[2.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[1.0, 2.0]),
        ClusteredLasso::new(0.1, 0.2),
    );
    for options in [
        Hybrid::<Vec<f64>>::new().max_iterations(0),
        Hybrid::new().cluster_sweeps(0),
        Hybrid::new().check_every(0),
        Hybrid::new().tolerance(0.0),
        Hybrid::new().tolerance(f64::NAN),
        Hybrid::new().step_policy(HybridStep::Backtracking {
            initial_curvature: Some(-1.0),
            max_backtracks: 10,
        }),
    ] {
        assert!(matches!(
            problem.fit_with(&options),
            Err(FitError::InvalidInput { .. })
        ));
    }
    assert!(matches!(
        problem.fit_with(
            &Hybrid::<Vec<f64>>::new().terminate_on(StoppingCriterion::duality_gap(1e-6))
        ),
        Err(FitError::UnsupportedCombination { .. })
    ));
    let runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::new(&x)),
        Box::new(Gaussian::new(&[1.0, 2.0])),
        Box::new(ClusteredLasso::new(0.1, 0.2)),
    );
    assert!(matches!(
        runtime.fit_with(&Hybrid::<Vec<f64>>::new()),
        Err(FitError::UnsupportedCombination { .. })
    ));
}
