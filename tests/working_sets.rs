//! Working-set restrictions must recover the complete problem's optimum.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, CoordinateDescent, Gaussian, Hybrid, HybridFit,
    Irls, L1, Lasso, Normalization, Problem, ProximalPenalty, Slope, SmoothDatafit, Termination,
    WorkingSetPolicy,
};

fn check_mapping<D: SmoothDatafit, P: ProximalPenalty>(
    rows: &[Vec<f64>],
    datafit: &D,
    penalty: &P,
    fit: &HybridFit,
) {
    let n = rows.len();
    let beta: Vec<_> = fit
        .coefficients()
        .iter()
        .enumerate()
        .map(|(j, b)| b * fit.preprocessing().scales().map_or(1.0, |s| s[j]))
        .collect();
    let eta: Vec<_> = rows
        .iter()
        .map(|row| {
            fit.intercept()
                + row
                    .iter()
                    .zip(fit.coefficients())
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
        })
        .collect();
    let objective = datafit.value(&eta).unwrap() + penalty.value(&beta).unwrap();
    assert!((objective - fit.objective()).abs() < 1e-11);
    let mut derivative = vec![0.0; n];
    datafit.gradient(&eta, &mut derivative).unwrap();
    let sum: f64 = derivative.iter().sum();
    let input: Vec<_> = beta
        .iter()
        .enumerate()
        .map(|(j, &b)| {
            let center = fit.preprocessing().centers().map_or(0.0, |c| c[j]);
            let scale = fit.preprocessing().scales().map_or(1.0, |s| s[j]);
            let gradient: f64 = rows
                .iter()
                .zip(&derivative)
                .map(|(row, d)| (row[j] - center) * d / scale)
                .sum();
            b - fit.step_size() * gradient
        })
        .collect();
    let mut output = vec![0.0; beta.len()];
    penalty.prox(&input, fit.step_size(), &mut output).unwrap();
    let mapping = beta
        .iter()
        .zip(output)
        .map(|(b, z)| (b - z).abs() / fit.step_size())
        .fold(sum.abs(), f64::max);
    assert!(mapping <= 1.01 * fit.stopping_threshold(), "{mapping}");
    assert!((mapping - fit.stopping_value()).abs() < 1e-12);
}

#[test]
fn hybrid_policies_preserve_complete_penalties_and_original_loss_checks() {
    let rows: Vec<Vec<f64>> = (0..16)
        .map(|i| {
            (0..12)
                .map(|j| {
                    let sign = if (i >> (j % 4)) & 1 == 0 { -1.0 } else { 1.0 };
                    sign * (1.0 + (i * 7 + j * 3) as f64 % 5.0 / 10.0)
                })
                .collect()
        })
        .collect();
    let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
    let x = Matrix::from_rows(&refs);
    let y: Vec<_> = rows
        .iter()
        .map(|row| 0.3 + 1.2 * row[0] - 0.8 * row[1] + 0.4 * row[2])
        .collect();
    let binary: Vec<_> = y.iter().map(|eta| 1.0 / (1.0 + (-eta).exp())).collect();
    macro_rules! compare {
        ($datafit:expr, $penalty:expr) => {{
            let datafit = $datafit;
            let penalty = $penalty;
            for normalization in [Normalization::None, Normalization::Standardize] {
                let problem = Problem::new(&x, datafit, penalty.clone()).normalize(normalization);
                let reference = problem
                    .fit_with(
                        &Hybrid::<Vec<f64>>::new()
                            .working_sets(false)
                            .screening(false)
                            .tolerance(1e-8)
                            .max_iterations(100_000),
                    )
                    .unwrap();
                assert_eq!(reference.termination(), Termination::Converged);
                for screening in [false, true] {
                    for policy in policies() {
                        let fit = problem
                            .fit_with(
                                &Hybrid::<Vec<f64>>::new()
                                    .working_set_policy(policy)
                                    .screening(screening)
                                    .tolerance(1e-8)
                                    .max_iterations(100_000),
                            )
                            .unwrap();
                        assert_eq!(fit.termination(), Termination::Converged, "{policy:?}");
                        assert!((fit.objective() - reference.objective()).abs() < 1e-9);
                        check_mapping(&rows, &datafit, &penalty, &fit);
                    }
                }
            }
        }};
    }
    compare!(Gaussian::new(&y), ClusteredLasso::new(0.08, 0.002));
    compare!(Gaussian::new(&y), ChainFusedLasso::new(0.08, 0.02));
    compare!(
        Gaussian::new(&y),
        Slope::new((0..12).map(|j| 0.1 - 0.003 * j as f64).collect())
    );
    compare!(Binomial::new(&binary), ClusteredLasso::new(0.01, 0.0005));
    compare!(Binomial::new(&binary), ChainFusedLasso::new(0.01, 0.005));
    compare!(
        Binomial::new(&binary),
        Slope::new((0..12).map(|j| 0.02 - 0.001 * j as f64).collect())
    );
}

#[test]
fn disabling_sets_and_setting_a_policy_use_the_last_setter() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let problem = Problem::new(&x, Gaussian::new(&[-2.0, 2.0]), L1::new(0.1));
    let explicit = WorkingSetPolicy::Adaptive {
        initial_size: 2,
        prune: false,
    };
    for solver in [
        CoordinateDescent::new()
            .working_set_policy(explicit)
            .working_sets(false),
        CoordinateDescent::new()
            .working_sets(true)
            .working_sets(false),
    ] {
        assert_eq!(
            problem
                .fit_with(&solver)
                .unwrap()
                .working_set_diagnostics()
                .policy,
            WorkingSetPolicy::Disabled
        );
    }
    let fit = problem
        .fit_with(
            &CoordinateDescent::new()
                .working_sets(false)
                .working_set_policy(explicit),
        )
        .unwrap();
    assert_eq!(fit.working_set_diagnostics().policy, explicit);
}

#[test]
fn zero_penalties_dense_solutions_and_iteration_limits_keep_full_diagnostics() {
    let x = Matrix::from_rows(&[&[1.0, 0.4], &[0.2, 1.0], &[-1.0, -0.3]]);
    let y = [2.0, -1.0, -2.0];
    for policy in policies() {
        let model = Lasso::new(0.0).working_set_policy(policy).tolerance(1e-10);
        let full = model.fit(&x, &y).unwrap();
        assert_eq!(full.termination(), Termination::Converged);
        assert!(full.kkt_violation() <= 1e-10);
        let limited = model.max_iterations(1).fit(&x, &y).unwrap();
        assert_eq!(limited.termination(), Termination::IterationLimit);
        assert!(limited.objective().is_finite());
        assert!(limited.kkt_violation() > limited.stopping_threshold());
        let empty = Lasso::new(0.1)
            .working_set_policy(policy)
            .fit(&Matrix::empty(3, 0), &y)
            .unwrap();
        assert_eq!(empty.termination(), Termination::Converged);
        assert_eq!(empty.working_set_diagnostics().active_dimension, 0);
    }
}

#[test]
fn restricted_sweeps_save_matrix_accesses_on_a_sparse_solution() {
    let rows: Vec<Vec<f64>> = (0..80)
        .map(|i| {
            (0..80)
                .map(|j| {
                    if i == j {
                        1.0
                    } else if i + 1 == j {
                        0.8
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
    let x = Matrix::from_rows(&refs);
    let y: Vec<_> = rows.iter().map(|row| 2.0 * row[0] + row[1]).collect();
    let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), L1::new(0.001))
        .normalize(Normalization::None);
    x.visits.set(0);
    let full = problem
        .fit_with(
            &CoordinateDescent::new()
                .working_sets(false)
                .tolerance(1e-12),
        )
        .unwrap();
    let full_visits = x.visits.get();
    x.visits.set(0);
    let restricted = problem
        .fit_with(&CoordinateDescent::new().tolerance(1e-12))
        .unwrap();
    assert_eq!(restricted.termination(), Termination::Converged);
    assert!((restricted.objective() - full.objective()).abs() < 1e-12);
    assert!(
        x.visits.get() < full_visits,
        "{} >= {full_visits}",
        x.visits.get()
    );
    assert!(restricted.working_set_diagnostics().peak_active_dimension < 80);
    assert!(restricted.working_set_diagnostics().full_checks < restricted.iterations());
}

fn policies() -> [WorkingSetPolicy; 4] {
    [
        WorkingSetPolicy::Disabled,
        WorkingSetPolicy::EverActive,
        WorkingSetPolicy::Adaptive {
            initial_size: 1,
            prune: false,
        },
        WorkingSetPolicy::Adaptive {
            initial_size: 1,
            prune: true,
        },
    ]
}

#[test]
fn omitted_coordinates_reenter_and_paths_resolve_their_own_default() {
    let x = Matrix::from_rows(&[
        &[-1.0, -1.0, -1.0],
        &[-1.0, 1.0, 1.0],
        &[1.0, -1.0, 1.0],
        &[1.0, 1.0, -1.0],
    ]);
    let y = [-6.0, 0.0, 2.0, 4.0];
    for policy in policies() {
        let model = Lasso::new(0.1).tolerance(1e-10).working_set_policy(policy);
        let fit = model.fit(&x, &y).unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        for (&actual, expected) in fit.coefficients().iter().zip([2.9, 1.9, 0.9]) {
            assert!((actual - expected).abs() < 1e-9);
        }
        assert!(fit.working_set_diagnostics().full_checks > 0);
    }
    let model = Lasso::new(0.1).tolerance(1e-10);
    assert_eq!(
        model.fit(&x, &y).unwrap().working_set_diagnostics().policy,
        WorkingSetPolicy::Adaptive {
            initial_size: 10,
            prune: true
        }
    );
    let path = model
        .fit_path(&x, &y, &[4.0, 0.1, 2.0, 0.01, 0.01])
        .unwrap();
    for fit in path.fits() {
        assert_eq!(fit.termination(), Termination::Converged);
        assert_eq!(
            fit.working_set_diagnostics().policy,
            WorkingSetPolicy::EverActive
        );
    }
    assert_eq!(path.fits()[4].iterations(), 0);
}

#[test]
fn coupled_zero_features_must_enter_together() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let y = [2.0, 2.0];
    let problem = Problem::new(
        &x,
        Gaussian::new(&y).fit_intercept(false),
        ChainFusedLasso::new(0.1, 10.0),
    )
    .normalize(Normalization::None);
    for policy in policies() {
        let fit = problem
            .fit_with(
                &Hybrid::<Vec<f64>>::new()
                    .screening(false)
                    .working_set_policy(policy)
                    .tolerance(1e-10),
            )
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        for &b in fit.coefficients() {
            assert!((b - 1.8).abs() < 1e-8);
        }
        if matches!(policy, WorkingSetPolicy::Adaptive { .. }) {
            assert!(fit.working_set_diagnostics().expansions > 0);
        }
    }
}

#[test]
fn irls_refreshes_excluded_features_with_the_weights() {
    let x = Matrix::from_rows(&[
        &[-1.0, -1.0, 0.0],
        &[-1.0, 1.0, 1.0],
        &[1.0, -1.0, 1.0],
        &[1.0, 1.0, 0.0],
        &[0.0, -1.0, -1.0],
        &[0.0, 1.0, -1.0],
    ]);
    let y = [0.1, 0.3, 0.6, 0.9, 0.2, 0.7];
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(0.02));
    let reference = problem
        .fit_with(&Irls::new().working_sets(false).tolerance(1e-9))
        .unwrap();
    for policy in policies() {
        let fit = problem
            .fit_with(&Irls::new().working_set_policy(policy).tolerance(1e-9))
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!((fit.objective() - reference.objective()).abs() < 1e-10);
        assert!(fit.kkt_violation() <= 1e-9);
    }
}

#[test]
fn invalid_budgets_are_rejected_even_for_empty_paths() {
    let x = Matrix::empty(2, 0);
    let problem = Problem::new(&x, Gaussian::new(&[0.0, 1.0]), L1::new(0.1));
    let policy = WorkingSetPolicy::Adaptive {
        initial_size: 0,
        prune: true,
    };
    assert!(
        problem
            .fit_with(&CoordinateDescent::new().working_set_policy(policy))
            .is_err()
    );
    assert!(
        problem
            .fit_path_with(&CoordinateDescent::new().working_set_policy(policy), vec![])
            .is_err()
    );
    assert!(
        problem
            .fit_with(&Irls::new().working_set_policy(policy))
            .is_err()
    );
}

#[test]
fn restricted_pg_splits_a_warm_started_nonzero_cluster() {
    use shrinkage::{InitialPoint, ObservedSolver};
    let rows: Vec<Vec<f64>> = (0..12)
        .map(|i| (0..12).map(|j| f64::from(i == j)).collect())
        .collect();
    let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
    let x = Matrix::from_rows(&refs);
    let mut y = [0.0; 12];
    y[0] = -4.0;
    y[1] = 4.0;
    let mut initial = [0.0; 12];
    initial[..2].fill(1.0);
    let problem = Problem::new(
        &x,
        Gaussian::new(&y).fit_intercept(false),
        ClusteredLasso::new(0.01, 0.001),
    )
    .normalize(Normalization::None);
    for prune in [false, true] {
        let solver = Hybrid::<Vec<f64>>::new()
            .screening(false)
            .tolerance(1e-10)
            .working_set_policy(WorkingSetPolicy::Adaptive {
                initial_size: 1,
                prune,
            });
        let fit = solver
            .solve_from_with_observer(
                &problem,
                Some(InitialPoint {
                    coefficients: &initial,
                    intercept: 0.0,
                }),
                &mut |_: shrinkage::SolverStep<'_>| {},
            )
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!((fit.coefficients()[0] + 3.748).abs() < 1e-8);
        assert!((fit.coefficients()[1] - 3.748).abs() < 1e-8);
        assert!(fit.coefficients()[2..].iter().all(|&b| b == 0.0));
        assert_ne!(fit.cluster_labels()[0], fit.cluster_labels()[1]);
        assert!(fit.working_set_diagnostics().peak_active_dimension < 12);
    }
}

#[test]
fn irls_paths_use_ever_active_history_with_caller_order() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [0.1, 0.3, 0.6, 0.9];
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(0.01));
    let solver = Irls::new().tolerance(1e-9);
    let path = problem
        .fit_path_with(
            &solver,
            vec![L1::new(0.01), L1::new(0.5), L1::new(0.02), L1::new(0.02)],
        )
        .unwrap();
    for (penalty, fit) in path.iter() {
        let cold = Problem::new(&x, Binomial::new(&y), *penalty)
            .fit_with(&solver)
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert_eq!(
            fit.working_set_diagnostics().policy,
            WorkingSetPolicy::EverActive
        );
        assert!((fit.objective() - cold.objective()).abs() < 1e-10);
    }
    assert_eq!(path.fits()[3].iterations(), 0);
}
