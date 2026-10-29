//! Hybrid paths share preparation while retaining independent fitted models.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Gaussian, Hybrid, Normalization, PathOptions,
    PathPenalty, PenaltyMaximum, Problem, Slope, Termination, WorkingSetPolicy,
};

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-6 * (1.0 + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

#[test]
fn explicit_gaussian_and_logistic_paths_match_cold_fits() {
    let x = Matrix::from_rows(&[
        &[-2.0, 1.0, 0.0],
        &[-1.0, 0.0, 2.0],
        &[0.0, -1.0, 1.0],
        &[1.0, 2.0, -1.0],
        &[2.0, -2.0, 0.0],
        &[3.0, 0.5, 1.0],
    ]);
    let y = [-2.0, -0.5, 1.0, 2.5, 2.0, 4.0];
    let binary = [0.1, 0.8, 0.3, 0.7, 0.2, 0.9];
    let solver = Hybrid::new().tolerance(1e-9).max_iterations(100_000);
    macro_rules! check {
        ($base:expr, $points:expr) => {
            for normalization in [Normalization::None, Normalization::default()] {
                let problem = Problem::new(&x, Gaussian::new(&y), $base).normalize(normalization);
                let path = problem.fit_path_with(&solver, $points).unwrap();
                assert_eq!(path.len(), 5);
                for (penalty, fit) in path.iter() {
                    let cold = Problem::new(&x, Gaussian::new(&y), penalty.clone())
                        .normalize(normalization)
                        .fit_with(&solver)
                        .unwrap();
                    assert_eq!(fit.termination(), Termination::Converged);
                    assert_eq!(
                        fit.working_set_diagnostics().policy,
                        WorkingSetPolicy::EverActive
                    );
                    close(fit.objective(), cold.objective());
                    for (a, b) in fit
                        .predict(&x)
                        .unwrap()
                        .into_iter()
                        .zip(cold.predict(&x).unwrap())
                    {
                        close(a, b);
                    }
                }
                let problem =
                    Problem::new(&x, Binomial::new(&binary), $base).normalize(normalization);
                let path = problem.fit_path_with(&solver, $points).unwrap();
                for (penalty, fit) in path.iter() {
                    let cold = Problem::new(&x, Binomial::new(&binary), penalty.clone())
                        .normalize(normalization)
                        .fit_with(&solver)
                        .unwrap();
                    assert_eq!(fit.termination(), Termination::Converged);
                    close(fit.objective(), cold.objective());
                    for (a, b) in fit
                        .predict(&x)
                        .unwrap()
                        .into_iter()
                        .zip(cold.predict(&x).unwrap())
                    {
                        close(a, b);
                    }
                }
            }
        };
    }
    check!(
        ClusteredLasso::new(1.0, 1.0),
        vec![
            ClusteredLasso::new(1.0, 2.0),
            ClusteredLasso::new(0.05, 0.01),
            ClusteredLasso::new(0.2, 0.5),
            ClusteredLasso::new(0.2, 0.5),
            ClusteredLasso::new(0.0, 0.0),
        ]
    );
    check!(
        ChainFusedLasso::new(1.0, 1.0),
        [2.0, 0.01, 0.5, 0.5, 0.0]
            .map(|s| ChainFusedLasso::new(0.1 * s, s).with_weights(vec![0.0, 2.0]))
            .to_vec()
    );
    check!(
        Slope::new(vec![1.0; 3]),
        [2.0, 0.01, 0.5, 0.5, 0.0]
            .map(|s| Slope::new(vec![s, 0.5 * s, 0.1 * s]))
            .to_vec()
    );
}

#[test]
fn empty_and_iteration_limited_paths_preserve_their_contract() {
    let x = Matrix::empty(3, 0);
    let y = [0.0, 1.0, 3.0];
    let solver = Hybrid::new();
    let problem = Problem::new(&x, Gaussian::new(&y), ClusteredLasso::new(1.0, 1.0));
    assert!(
        problem
            .fit_path_with(&solver, Vec::<ClusteredLasso>::new())
            .unwrap()
            .is_empty()
    );
    let path = problem
        .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(2))
        .unwrap();
    assert!(path.fits().iter().all(|fit| fit.coefficients().is_empty()));
    let x = Matrix::from_rows(&[&[-2.0, 1.0], &[0.0, -1.0], &[2.0, 1.0]]);
    let problem = Problem::new(&x, Gaussian::new(&y), Slope::new(vec![0.01, 0.005]));
    let path = problem
        .fit_path_with(
            &Hybrid::new()
                .max_iterations(1)
                .tolerance(1e-15)
                .step_policy(shrinkage::HybridStep::Backtracking {
                    initial_curvature: Some(100.0),
                    max_backtracks: 100,
                }),
            vec![Slope::new(vec![0.01, 0.005]); 2],
        )
        .unwrap();
    assert_eq!(path.fits()[0].termination(), Termination::IterationLimit);
    assert!(path.fits()[1].objective() < path.fits()[0].objective());
}

#[test]
fn every_automatic_hybrid_combination_matches_cold_proximal_fits() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let solver = Hybrid::new().tolerance(1e-9).max_iterations(100_000);
    let pg = shrinkage::ProximalGradient::new()
        .tolerance(1e-9)
        .max_iterations(100_000);
    let options = PathOptions::new()
        .nlambda(6)
        .lambda_min_ratio(0.05)
        .early_stopping(false);
    macro_rules! check {
        ($datafit:expr, $penalty:expr) => {{
            let datafit = $datafit;
            let problem = Problem::new(&x, datafit, $penalty);
            let path = problem
                .fit_regularization_path_with(&solver, &options)
                .unwrap();
            assert_eq!(path.len(), 6);
            assert!(
                path.fits()[0]
                    .coefficients()
                    .iter()
                    .all(|&b| b.abs() < 1e-12)
            );
            for ((penalty, fit), &deviance) in path.iter().zip(path.deviances().unwrap()) {
                let cold = Problem::new(&x, datafit, penalty.clone())
                    .fit_with(&pg)
                    .unwrap();
                assert_eq!(fit.termination(), Termination::Converged);
                close(fit.objective(), cold.objective());
                let predicted = fit.predict(&x).unwrap();
                for (a, b) in predicted.iter().zip(cold.predict(&x).unwrap()) {
                    close(*a, b);
                }
                use shrinkage::NullModelDatafit;
                close(deviance, datafit.deviance(&predicted).unwrap().unwrap());
            }
        }};
    }
    for intercept in [false, true] {
        let y = [-1.0, -7.0, 9.0, 3.0];
        check!(
            Gaussian::new(&y).fit_intercept(intercept),
            ClusteredLasso::new(1.0, 0.3)
        );
        check!(
            Gaussian::new(&y).fit_intercept(intercept),
            ChainFusedLasso::new(1.0, 0.3).with_weights(vec![2.0])
        );
        check!(
            Gaussian::new(&y).fit_intercept(intercept),
            Slope::new(vec![1.0, 0.5])
        );
        let y = [0.1, 0.4, 0.6, 0.9];
        check!(
            Binomial::new(&y).fit_intercept(intercept),
            ClusteredLasso::new(1.0, 0.3)
        );
        check!(
            Binomial::new(&y).fit_intercept(intercept),
            ChainFusedLasso::new(1.0, 0.3).with_weights(vec![2.0])
        );
        check!(
            Binomial::new(&y).fit_intercept(intercept),
            Slope::new(vec![1.0, 0.5])
        );
    }
}

#[test]
fn coupled_thresholds_match_independent_cut_constraints() {
    fn update(maximum: &mut f64, sum: f64, capacity: f64) {
        if capacity == 0.0 {
            if sum != 0.0 {
                *maximum = f64::INFINITY;
            }
        } else {
            *maximum = maximum.max(sum.abs() / capacity);
        }
    }
    for gradient in [
        vec![5.0, -3.0],
        vec![1.0, -2.0, 1.0],
        vec![2.0, -2.0, 3.0, -3.0],
        vec![0.0; 4],
    ] {
        let p = gradient.len();
        for a in [0.0, 0.2, 2.0] {
            for b in [0.0, 0.7, 2.0] {
                let mut expected = 0.0_f64;
                for mask in 1_usize..1 << p {
                    let count = mask.count_ones() as usize;
                    let sum = gradient
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| mask & (1 << j) != 0)
                        .map(|(_, &g)| g)
                        .sum::<f64>();
                    update(
                        &mut expected,
                        sum,
                        a * count as f64 + b * (count * (p - count)) as f64,
                    );
                }
                let actual = ClusteredLasso::new(a, b)
                    .maximum_penalty(&gradient)
                    .unwrap();
                if expected.is_infinite() {
                    assert_eq!(actual, PenaltyMaximum::Unbounded);
                } else {
                    close(actual.finite().unwrap(), expected);
                }
                for weights in [
                    vec![1.0; p - 1],
                    (0..p - 1).map(|j| if j == 0 { 0.0 } else { 2.0 }).collect(),
                ] {
                    let mut expected = 0.0_f64;
                    for start in 0..p {
                        for end in start + 1..=p {
                            let left = if start == 0 { 0.0 } else { weights[start - 1] };
                            let right = if end == p { 0.0 } else { weights[end - 1] };
                            update(
                                &mut expected,
                                gradient[start..end].iter().sum(),
                                a * (end - start) as f64 + b * (left + right),
                            );
                        }
                    }
                    let actual = ChainFusedLasso::new(a, b)
                        .with_weights(weights)
                        .maximum_penalty(&gradient)
                        .unwrap();
                    if expected.is_infinite() {
                        assert_eq!(actual, PenaltyMaximum::Unbounded);
                    } else {
                        close(actual.finite().unwrap(), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn unused_complete_graph_edges_do_not_overflow_a_path_threshold() {
    let maximum = ClusteredLasso::new(1.0, f64::MAX)
        .maximum_penalty(&[1.0, -1.0])
        .unwrap()
        .finite()
        .unwrap();
    assert!(maximum > 0.0);
    assert!((maximum * f64::MAX - 1.0).abs() < 1e-14);
}

#[test]
fn screening_and_working_sets_reactivate_features_at_new_strengths() {
    let rows: Vec<Vec<f64>> = (0..8)
        .map(|i| {
            vec![
                if i & 1 == 0 { -1.0 } else { 1.0 },
                (if i & 2 == 0 { -0.2 } else { 0.2 }) + (if i & 1 == 0 { -0.1 } else { 0.1 }),
                (if i & 4 == 0 { -0.1 } else { 0.1 }) + (if i & 1 == 0 { -0.02 } else { 0.02 }),
                0.0,
            ]
        })
        .collect();
    let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let y: Vec<_> = rows
        .iter()
        .map(|r| 0.3 + 3.0 * r[0] + 2.0 * r[1] + 4.0 * r[2])
        .collect();
    let binary: Vec<_> = y.iter().map(|eta| 1.0 / (1.0 + (-eta).exp())).collect();
    macro_rules! verify {
        ($datafit:expr, $base:expr) => {{
            let datafit = $datafit;
            let base = $base;
            let points = || {
                [1.0, 0.001, 1.0, 0.0]
                    .map(|lambda| base.at_strength(lambda).unwrap())
                    .to_vec()
            };
            let problem = Problem::new(&x, datafit, base.clone()).normalize(Normalization::None);
            let solver = Hybrid::new()
                .tolerance(1e-9)
                .max_iterations(100_000)
                .step_policy(shrinkage::HybridStep::FixedBound)
                .check_every(1);
            let reference = problem
                .fit_path_with(
                    &solver.clone().screening(false).working_sets(false),
                    points(),
                )
                .unwrap();
            for screening in [true, false] {
                for policy in [
                    WorkingSetPolicy::Disabled,
                    WorkingSetPolicy::EverActive,
                    WorkingSetPolicy::Adaptive {
                        initial_size: 1,
                        prune: true,
                    },
                ] {
                    let path = problem
                        .fit_path_with(
                            &solver
                                .clone()
                                .screening(screening)
                                .working_set_policy(policy),
                            points(),
                        )
                        .unwrap();
                    for (fit, cold) in path.fits().iter().zip(reference.fits()) {
                        assert_eq!(fit.termination(), Termination::Converged);
                        close(fit.objective(), cold.objective());
                        for (a, b) in fit
                            .predict(&x)
                            .unwrap()
                            .iter()
                            .zip(cold.predict(&x).unwrap())
                        {
                            close(*a, b);
                        }
                    }
                    if screening && policy == WorkingSetPolicy::Disabled {
                        assert!(
                            path.fits()[0].screening_diagnostics().screened_features > 0,
                            "{policy:?}: {:?}",
                            path.fits()[0].screening_diagnostics()
                        );
                    }
                    assert!(path.fits()[1].coefficients()[2].abs() > 1.0);
                    assert_eq!(path.fits()[3].screening_diagnostics().screened_features, 0);
                }
            }
        }};
    }
    verify!(Gaussian::new(&y), ClusteredLasso::new(0.3, 0.001));
    verify!(Gaussian::new(&y), ChainFusedLasso::new(0.3, 0.001));
    verify!(Gaussian::new(&y), Slope::new(vec![0.3, 0.29, 0.28, 0.27]));
    verify!(Binomial::new(&binary), ClusteredLasso::new(0.05, 0.0001));
    verify!(Binomial::new(&binary), ChainFusedLasso::new(0.05, 0.0001));
    verify!(
        Binomial::new(&binary),
        Slope::new(vec![0.05, 0.049, 0.048, 0.047])
    );
}

#[test]
fn boundary_null_models_and_invalid_points_fail_without_partial_results() {
    use shrinkage::FitError;
    let x = Matrix::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0], &[1.0, 0.0]]);
    let solver = Hybrid::new().tolerance(1e-8);
    let boundary = Problem::new(&x, Binomial::new(&[1.0; 3]), ClusteredLasso::new(1.0, 1.0));
    assert!(matches!(
        boundary.fit_regularization_path_with(&solver, &PathOptions::new()),
        Err(FitError::InvalidInput { .. })
    ));
    let fixed = Problem::new(
        &x,
        Binomial::new(&[1.0; 3]).fit_intercept(false),
        ClusteredLasso::new(1.0, 1.0),
    );
    assert!(
        fixed
            .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(2))
            .is_ok()
    );
    assert!(
        boundary
            .fit_path_with(&solver, vec![ClusteredLasso::new(1.0, 1.0)])
            .is_ok()
    );
    let problem = Problem::new(
        &x,
        Gaussian::new(&[1.0, 2.0, 4.0]),
        ChainFusedLasso::new(1.0, 1.0),
    );
    assert!(matches!(
        problem.fit_path_with(
            &solver,
            vec![
                ChainFusedLasso::new(1.0, 1.0),
                ChainFusedLasso::new(-1.0, 1.0)
            ]
        ),
        Err(FitError::InvalidInput { .. })
    ));
    assert!(matches!(
        problem.fit_path_with(
            &solver,
            vec![ChainFusedLasso::new(1.0, 1.0).with_weights(vec![])]
        ),
        Err(FitError::InvalidInput { .. })
    ));
}

#[test]
fn automatic_hybrid_paths_scale_the_complete_penalty() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-1.0, -7.0, 9.0, 3.0];
    let options = PathOptions::new().nlambda(4).early_stopping(false);
    let solver = Hybrid::new().tolerance(1e-9);
    let problem = Problem::new(&x, Gaussian::new(&y), ClusteredLasso::new(2.0, 1.0));
    let path = problem
        .fit_regularization_path_with(&solver, &options)
        .unwrap();
    close(path.lambda_start().unwrap(), 5.0 / 3.0);
    assert_eq!(path.fits()[0].coefficients(), &[0.0, 0.0]);
    for ((penalty, fit), &lambda) in path.iter().zip(path.lambdas().unwrap()) {
        close(penalty.l1(), 2.0 * lambda);
        close(penalty.fusion(), lambda);
        let cold = Problem::new(&x, Gaussian::new(&y), penalty.clone())
            .fit_with(&solver)
            .unwrap();
        close(fit.objective(), cold.objective());
    }
    let gradient = [5.0, -3.0];
    assert_eq!(
        ClusteredLasso::new(0.0, 1.0)
            .maximum_penalty(&gradient)
            .unwrap(),
        PenaltyMaximum::Unbounded
    );
    assert!(
        Problem::new(&x, Gaussian::new(&y), ChainFusedLasso::new(0.0, 1.0))
            .fit_regularization_path_with(&solver, &options)
            .is_err()
    );
    assert!(
        Problem::new(&x, Gaussian::new(&y), ChainFusedLasso::new(0.0, 1.0))
            .fit_regularization_path_with(&solver, &options.lambda_start(1.0))
            .is_ok()
    );
}
