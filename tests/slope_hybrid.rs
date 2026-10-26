//! Gaussian SLOPE hybrid fits and optimization-scale magnitude clusters.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use serde_json::Value;
use shrinkage::{
    ChainFusedLasso, ClusteredLasso, FitError, Gaussian, Hybrid, HybridStep, MatrixDesign,
    Normalization, Problem, ProximalGradient, RuntimeProblem, Slope, StoppingCriterion,
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

fn sorted_penalty(beta: &[f64], weights: &[f64]) -> f64 {
    let mut sorted: Vec<_> = beta.iter().map(|b| b.abs()).collect();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    sorted.iter().zip(weights).map(|(b, w)| b * w).sum()
}

fn mapping(
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    free_intercept: bool,
    penalty: &Slope,
    step: f64,
) -> f64 {
    use shrinkage::ProximalPenalty;
    let residual: Vec<_> = rows
        .iter()
        .zip(y)
        .map(|(row, y)| row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>() + intercept - y)
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
    let intercept_mapping = if free_intercept {
        residual.iter().sum::<f64>().abs() / y.len() as f64
    } else {
        0.0
    };
    beta.iter()
        .zip(output)
        .map(|(b, z)| (b - z).abs() / step)
        .fold(intercept_mapping, f64::max)
}

#[test]
fn orthogonal_fit_retains_opposite_signs_in_one_magnitude_cluster() {
    let x = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 2.0]]);
    let fit = Problem::new(
        &x,
        Gaussian::new(&[-3.0, 3.0]).fit_intercept(false),
        Slope::new(vec![3.0, 1.0]),
    )
    .normalize(Normalization::None)
    .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(fit.coefficients(), [-0.5, 0.5]);
    assert_eq!(fit.cluster_labels(), [0, 0]);
    assert_eq!(fit.cluster_values(), [0.5]);
    assert_eq!(fit.coefficient_signs(), [-1.0, 1.0]);
}

#[test]
fn hybrid_and_pg_match_independent_qp_fixtures_and_certificates() {
    let fixtures: Value = serde_json::from_str(include_str!("slope/reference.json")).unwrap();
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
        let free_intercept = case["fit_intercept"].as_bool().unwrap();
        let penalty = Slope::new(weights.clone());
        let problem = Problem::new(
            &x,
            Gaussian::new(&y).fit_intercept(free_intercept),
            penalty.clone(),
        )
        .normalize(Normalization::None);
        let pg = problem
            .fit_with(
                &ProximalGradient::<Vec<f64>>::new()
                    .tolerance(1e-9)
                    .max_iterations(100_000),
            )
            .unwrap();
        assert_eq!(pg.termination(), Termination::Converged, "{}", case["name"]);
        close(pg.objective(), case["objective"].as_f64().unwrap(), 1e-8);
        for policy in [HybridStep::default(), HybridStep::FixedBound] {
            let fit = problem
                .fit_with(
                    &Hybrid::<Vec<f64>>::new()
                        .tolerance(1e-9)
                        .max_iterations(100_000)
                        .step_policy(policy),
                )
                .unwrap();
            assert_eq!(
                fit.termination(),
                Termination::Converged,
                "{}: {}",
                case["name"],
                fit.stopping_value()
            );
            close(fit.objective(), case["objective"].as_f64().unwrap(), 1e-8);
            close(fit.objective(), pg.objective(), 1e-10);
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
                2e-12,
            );
            assert!(fit.stopping_value() <= fit.stopping_threshold());
            assert_eq!(fit.iterations(), fit.pg_steps() + fit.coordinate_sweeps());
            assert!(
                fit.cluster_values()
                    .windows(2)
                    .all(|pair| pair[0] > pair[1])
            );
            for (j, &label) in fit.cluster_labels().iter().enumerate() {
                assert_eq!(
                    fit.coefficients()[j],
                    fit.coefficient_signs()[j] * fit.cluster_values()[label]
                );
            }
            for (&actual, expected) in fit.predict(&x).unwrap().iter().zip(vector(&case["fitted"]))
            {
                close(actual, expected, 5e-6);
            }
        }

        // Dual prefix feasibility and equality in the support inequality certify
        // the oracle's subgradient without Shrinkage's value or proximal map.
        let beta = vector(&case["coefficients"]);
        let intercept = case["intercept"].as_f64().unwrap();
        let subgradient = vector(&case["subgradient"]);
        let residual: Vec<_> = rows
            .iter()
            .zip(&y)
            .map(|(row, y)| row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>() + intercept - y)
            .collect();
        let mut sorted: Vec<_> = subgradient.iter().map(|g| g.abs()).collect();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let (mut gradient_sum, mut weight_sum) = (0.0, 0.0);
        for (&g, &weight) in sorted.iter().zip(&weights) {
            gradient_sum += g;
            weight_sum += weight;
            assert!(gradient_sum <= weight_sum + 1e-8);
        }
        close(
            beta.iter().zip(&subgradient).map(|(b, g)| b * g).sum(),
            sorted_penalty(&beta, &weights),
            1e-8,
        );
        for (j, &g) in subgradient.iter().enumerate() {
            close(
                rows.iter()
                    .zip(&residual)
                    .map(|(row, r)| row[j] * r / y.len() as f64)
                    .sum::<f64>()
                    + g,
                0.0,
                1e-8,
            );
        }
        if free_intercept {
            close(residual.iter().sum::<f64>() / y.len() as f64, 0.0, 1e-8);
        }
        close(
            residual
                .iter()
                .map(|r| r * r / (2.0 * y.len() as f64))
                .sum::<f64>()
                + sorted_penalty(&beta, &weights),
            case["objective"].as_f64().unwrap(),
            1e-12,
        );
    }
}

#[test]
fn explicit_and_lazy_normalization_preserve_predictions_and_magnitude_clusters() {
    let rows = [
        vec![1.0, 2.0, -1.0],
        vec![2.0, -1.0, 0.0],
        vec![3.0, 4.0, 1.0],
        vec![-2.0, 1.0, 2.0],
        vec![1.0, 0.0, 0.0],
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
            for supplied in [false, true] {
                let penalty = Slope::new(vec![0.3, 0.1, 0.0]);
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
                    penalty,
                )
                .normalize(Normalization::None);
                let reference = reference_problem
                    .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
                    .unwrap();
                let pg = reference_problem
                    .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-10))
                    .unwrap();
                assert_eq!(reference.termination(), Termination::Converged);
                assert_eq!(pg.termination(), Termination::Converged);
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
                        fit.coefficient_signs()[j] * fit.cluster_values()[label],
                        1e-14,
                    );
                }
                close(
                    fit.intercept(),
                    reference.intercept()
                        - centers
                            .iter()
                            .zip(fit.coefficients())
                            .map(|(c, b)| c * b)
                            .sum::<f64>(),
                    2e-8,
                );
                assert_eq!(fit.preprocessing().centers_were_supplied(), supplied);
                assert_eq!(fit.preprocessing().scales_were_supplied(), supplied);
            }
        }
    }
}

#[test]
fn normalization_can_change_original_scale_magnitude_equalities() {
    let x = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 4.0]]);
    let fit = Problem::new(
        &x,
        Gaussian::new(&[-3.0, 3.0]).fit_intercept(false),
        Slope::new(vec![3.0, 1.0]),
    )
    .normalize(Normalization::None)
    .with_scales(vec![1.0, 2.0])
    .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
    .unwrap();
    assert_eq!(fit.cluster_labels(), [0, 0]);
    assert_eq!(fit.cluster_values(), [0.5]);
    assert_eq!(fit.coefficients(), [-0.5, 0.25]);
    assert_eq!(fit.coefficient_signs(), [-1.0, 1.0]);
}

#[test]
fn empty_features_zero_design_and_single_feature_limits() {
    for free_intercept in [false, true] {
        let x = Matrix::empty(3, 0);
        let fit = Problem::new(
            &x,
            Gaussian::new(&[1.0, 2.0, 3.0]).fit_intercept(free_intercept),
            Slope::new(vec![]),
        )
        .fit_with(&Hybrid::<Vec<f64>>::new())
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!(fit.cluster_values().is_empty());
        assert!(fit.coefficient_signs().is_empty());
        assert_eq!(fit.intercept(), if free_intercept { 2.0 } else { 0.0 });
        let zero = Matrix::empty(3, 2);
        let fit = Problem::new(
            &zero,
            Gaussian::new(&[1.0, 2.0, 3.0]).fit_intercept(free_intercept),
            Slope::new(vec![0.1, 0.0]),
        )
        .fit_with(&Hybrid::<Vec<f64>>::new())
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert_eq!(fit.cluster_values(), [0.0]);
        assert_eq!(fit.coefficient_signs(), [0.0; 2]);
        assert!(fit.predict(&Matrix::empty(0, 2)).unwrap().is_empty());
    }
    let x = Matrix::from_rows(&[&[1.0], &[-1.0]]);
    for weight in [0.0, 0.5, 3.0] {
        let fit = Problem::new(
            &x,
            Gaussian::new(&[-2.0, 2.0]).fit_intercept(false),
            Slope::new(vec![weight]),
        )
        .normalize(Normalization::None)
        .fit_with(&Hybrid::<Vec<f64>>::new())
        .unwrap();
        close(fit.coefficients()[0], -(2.0_f64 - weight).max(0.0), 1e-12);
    }
}

#[test]
fn budgets_and_diagnostics_preserve_the_committed_schedule() {
    let x = Matrix::from_rows(&[
        &[1.0, 2.0, -1.0],
        &[2.0, -1.0, 0.0],
        &[3.0, 4.0, 1.0],
        &[-2.0, 1.0, 2.0],
        &[1.0, 0.0, 0.0],
    ]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[3.0, -1.0, 4.0, 0.0, 2.0]),
        Slope::new(vec![0.3, 0.1, 0.0]),
    )
    .normalize(Normalization::None);
    for sweeps in [1, 5] {
        for budget in 1..=9 {
            let options = Hybrid::<Vec<f64>>::new()
                .tolerance(1e-30)
                .cluster_sweeps(sweeps)
                .max_iterations(budget)
                .step_policy(HybridStep::FixedBound);
            let fit = problem.fit_with(&options.clone().check_every(1)).unwrap();
            let sparse_checks = problem.fit_with(&options.check_every(4)).unwrap();
            assert_eq!(fit.termination(), Termination::IterationLimit);
            assert_eq!(fit.pg_steps(), (budget - 1) / (sweeps + 1) + 1);
            assert_eq!(fit.coordinate_sweeps(), budget - fit.pg_steps());
            assert_eq!(fit.iterations(), budget);
            assert_eq!(fit.coefficients(), sparse_checks.coefficients());
            assert_eq!(fit.intercept(), sparse_checks.intercept());
            assert_eq!(fit.objective(), sparse_checks.objective());
            assert_eq!(fit.stopping_value(), sparse_checks.stopping_value());
            assert_eq!(
                sparse_checks.diagnostic_checks(),
                1 + budget / 4 + usize::from(budget % 4 != 0)
            );
        }
    }
}

#[test]
fn invalid_inputs_fail_before_preparation_and_runtime_selection_stays_unsupported() {
    let x = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, -1.0]]);
    for weights in [
        vec![],
        vec![1.0],
        vec![1.0; 3],
        vec![1.0, 2.0],
        vec![1.0, -1.0],
        vec![f64::NAN, 0.0],
    ] {
        assert!(matches!(
            Problem::new(&x, Gaussian::new(&[1.0, 2.0]), Slope::new(weights))
                .fit_with(&Hybrid::<Vec<f64>>::new()),
            Err(FitError::InvalidInput { .. })
        ));
        assert_eq!(x.visits.get(), 0);
    }
    let problem = Problem::new(&x, Gaussian::new(&[1.0, 2.0]), Slope::new(vec![0.2, 0.1]));
    for solver in [
        Hybrid::new().cluster_sweeps(0),
        Hybrid::new().max_iterations(0),
        Hybrid::new().check_every(0),
        Hybrid::new().tolerance(0.0),
        Hybrid::new().tolerance(f64::NAN),
    ] {
        assert!(matches!(
            problem.fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
        assert_eq!(x.visits.get(), 0);
    }
    assert!(matches!(
        problem.fit_with(
            &Hybrid::<Vec<f64>>::new().terminate_on(StoppingCriterion::kkt_violation(1e-6))
        ),
        Err(FitError::UnsupportedCombination { .. })
    ));
    let runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
        Box::new(Gaussian::new(&[1.0, 2.0])),
        Box::new(Slope::new(vec![0.2, 0.1])),
    );
    assert!(matches!(
        runtime.fit_with(&Hybrid::<Vec<f64>>::new()),
        Err(FitError::UnsupportedCombination { .. })
    ));
    assert_eq!(x.visits.get(), 0);
}

#[test]
fn sign_accessor_preserves_existing_signed_cluster_values() {
    let x = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 2.0]]);
    let y = [-3.0, 3.0];
    let clustered = Problem::new(
        &x,
        Gaussian::new(&y).fit_intercept(false),
        ClusteredLasso::new(0.1, 0.1),
    )
    .normalize(Normalization::None)
    .fit_with(&Hybrid::<Vec<f64>>::new())
    .unwrap();
    let chain = Problem::new(
        &x,
        Gaussian::new(&y).fit_intercept(false),
        ChainFusedLasso::new(0.1, 0.1),
    )
    .normalize(Normalization::None)
    .fit_with(&Hybrid::<Vec<f64>>::new())
    .unwrap();
    for fit in [clustered, chain] {
        assert_eq!(fit.coefficient_signs(), [-1.0, 1.0]);
        for (j, &label) in fit.cluster_labels().iter().enumerate() {
            assert_eq!(fit.coefficients()[j], fit.cluster_values()[label]);
        }
    }
}
