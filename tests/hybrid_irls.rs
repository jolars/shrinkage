//! Logistic hybrids checked against independent optimality equations and PG.

#[path = "common/matrix.rs"]
mod matrix;

use std::convert::Infallible;

use matrix::Matrix;
use shrinkage::lazymatrix::{RawColumn, RawColumns};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Datafit, FitError, Hybrid, HybridFit, HybridStep,
    Normalization, Problem, ProximalGradient, ProximalPenalty, Slope, Solver, Termination,
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn check<P: ProximalPenalty>(fit: &HybridFit, x: &Matrix, y: &[f64], penalty: &P, intercept: bool) {
    let eta = fit.predict(x).unwrap();
    let beta: Vec<_> = fit
        .coefficients()
        .iter()
        .enumerate()
        .map(|(j, b)| b * fit.preprocessing().scales().map_or(1.0, |s| s[j]))
        .collect();
    close(
        fit.objective(),
        Binomial::new(y).value(&eta).unwrap() + penalty.value(&beta).unwrap(),
        2e-13,
    );
    assert_eq!(fit.iterations(), fit.pg_steps() + fit.coordinate_sweeps());
    let derivative: Vec<_> = eta
        .iter()
        .zip(y)
        .map(|(&eta, &target)| {
            let tail = (-eta.abs()).exp();
            (if eta >= 0.0 {
                (1.0 - target) - tail / (1.0 + tail)
            } else {
                tail / (1.0 + tail) - target
            }) / y.len() as f64
        })
        .collect();
    let sum: f64 = derivative.iter().sum();
    let input: Vec<_> = beta
        .iter()
        .enumerate()
        .map(|(j, &b)| {
            let mut g = 0.0;
            x.raw_column(j)
                .for_each_stored(|i, a| g += a * derivative[i]);
            let center = fit.preprocessing().centers().map_or(0.0, |v| v[j]);
            let scale = fit.preprocessing().scales().map_or(1.0, |v| v[j]);
            b - fit.step_size() * (g - center * sum) / scale
        })
        .collect();
    let mut prox = vec![0.0; beta.len()];
    penalty.prox(&input, fit.step_size(), &mut prox).unwrap();
    let mapping = beta
        .iter()
        .zip(prox)
        .map(|(b, z)| (b - z).abs() / fit.step_size())
        .fold(if intercept { sum.abs() } else { 0.0 }, f64::max);
    close(fit.stopping_value(), mapping, 3e-13);
    if fit.termination() == Termination::Converged {
        assert!(fit.stopping_value() <= fit.stopping_threshold());
    }
    for (j, &label) in fit.cluster_labels().iter().enumerate() {
        let value = fit.cluster_values()[label];
        close(beta[j].abs(), value.abs(), 1e-13);
        assert_eq!(
            fit.coefficient_signs()[j],
            beta[j].signum() * f64::from(beta[j] != 0.0)
        );
    }
}

fn compare<P>(penalty: P)
where
    P: ProximalPenalty + Clone,
    for<'a> Hybrid<Vec<f64>>:
        Solver<Problem<&'a Matrix, Binomial<'a>, P>, Fit = HybridFit, Error = FitError<Infallible>>,
{
    let x = Matrix::from_rows(&[
        &[1.0, 2.0, -1.0, 0.0],
        &[0.0, 1.0, 2.0, 0.0],
        &[2.0, 0.0, 1.0, 0.0],
        &[1.0, -1.0, 1.0, 0.0],
        &[3.0, 0.5, -1.0, 0.0],
        &[-1.0, 1.0, -2.0, 0.0],
    ]);
    let y = [0.1, 0.8, 0.7, 0.2, 0.9, 0.3];
    for intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Center,
            Normalization::Standardize,
        ] {
            let problem = Problem::new(
                &x,
                Binomial::new(&y).fit_intercept(intercept),
                penalty.clone(),
            )
            .normalize(normalization);
            let reference = problem
                .fit_with(
                    &ProximalGradient::<Vec<f64>>::new()
                        .tolerance(1e-10)
                        .max_iterations(100_000),
                )
                .unwrap();
            assert_eq!(reference.termination(), Termination::Converged);
            for policy in [HybridStep::default(), HybridStep::FixedBound] {
                let fit = problem
                    .fit_with(
                        &Hybrid::<Vec<f64>>::new()
                            .tolerance(1e-9)
                            .step_policy(policy),
                    )
                    .unwrap();
                assert_eq!(fit.termination(), Termination::Converged);
                assert!(fit.outer_iterations() > 0);
                check(&fit, &x, &y, &penalty, intercept);
                close(fit.objective(), reference.objective(), 2e-10);
                for (actual, expected) in fit
                    .predict(&x)
                    .unwrap()
                    .into_iter()
                    .zip(reference.predict(&x).unwrap())
                {
                    close(actual, expected, 2e-6);
                }
            }
        }
    }
}

#[test]
fn all_penalties_match_original_logistic_pg() {
    compare(ClusteredLasso::new(0.015, 0.01));
    compare(ClusteredLasso::new(0.0, 0.01));
    compare(ClusteredLasso::new(0.0, 0.0));
    compare(ChainFusedLasso::new(0.015, 0.01).with_weights(vec![0.0, 2.0, 0.5]));
    compare(ChainFusedLasso::new(0.0, 0.01));
    compare(ChainFusedLasso::new(0.0, 0.0));
    compare(Slope::new(vec![0.04, 0.025, 0.01, 0.0]));
    compare(Slope::new(vec![0.02; 4]));
    compare(Slope::new(vec![0.0; 4]));
}

fn supplied_normalization<P>(penalty: P)
where
    P: ProximalPenalty + Clone,
    for<'a> Hybrid<Vec<f64>>:
        Solver<Problem<&'a Matrix, Binomial<'a>, P>, Fit = HybridFit, Error = FitError<Infallible>>,
{
    let rows = [
        [1.0, 2.0, -1.0],
        [0.0, 1.0, 2.0],
        [2.0, 0.0, 1.0],
        [1.0, -1.0, 1.0],
        [3.0, 0.5, -1.0],
    ];
    let centers = [0.5, -0.25, 0.1];
    let scales = [2.0, 3.0, 0.5];
    let normalized: Vec<Vec<_>> = rows
        .iter()
        .map(|row| (0..3).map(|j| (row[j] - centers[j]) / scales[j]).collect())
        .collect();
    let x = Matrix::from_rows(&rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>());
    let explicit = Matrix::from_rows(&normalized.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let y = [0.1, 0.8, 0.7, 0.2, 0.9];
    for intercept in [false, true] {
        let solver = Hybrid::<Vec<f64>>::new().tolerance(1e-10);
        let fit = Problem::new(
            &x,
            Binomial::new(&y).fit_intercept(intercept),
            penalty.clone(),
        )
        .normalize(Normalization::None)
        .with_centers(centers.to_vec())
        .with_scales(scales.to_vec())
        .fit_with(&solver)
        .unwrap();
        let reference = Problem::new(
            &explicit,
            Binomial::new(&y).fit_intercept(intercept),
            penalty.clone(),
        )
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert_eq!(reference.termination(), Termination::Converged);
        check(&fit, &x, &y, &penalty, intercept);
        close(fit.objective(), reference.objective(), 1e-12);
        for (a, b) in fit
            .predict(&x)
            .unwrap()
            .into_iter()
            .zip(reference.predict(&explicit).unwrap())
        {
            close(a, b, 2e-8);
        }
        for (j, &b) in fit.coefficients().iter().enumerate() {
            close(b * scales[j], reference.coefficients()[j], 2e-8);
        }
    }
}

#[test]
fn supplied_and_explicit_normalization_preserve_logistic_predictions() {
    supplied_normalization(ClusteredLasso::new(0.015, 0.01));
    supplied_normalization(ChainFusedLasso::new(0.015, 0.01));
    supplied_normalization(Slope::new(vec![0.03, 0.02, 0.0]));
}

#[test]
fn scalar_fit_satisfies_analytic_stationarity() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let fit = Problem::new(
        &x,
        Binomial::new(&[0.0, 1.0]).fit_intercept(false),
        ClusteredLasso::new(0.2, 0.7),
    )
    .normalize(Normalization::None)
    .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-11))
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.coefficients()[0], 4.0_f64.ln(), 1e-9);
}

#[test]
fn limits_keep_fresh_logistic_diagnostics_and_separate_work() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [0.0, 1.0];
    let penalty = ClusteredLasso::new(0.02, 0.0);
    let problem = Problem::new(&x, Binomial::new(&y).fit_intercept(false), penalty.clone())
        .normalize(Normalization::None);
    let solver = Hybrid::<Vec<f64>>::new().tolerance(1e-12);
    for options in [
        solver.clone().max_iterations(1),
        solver.clone().max_outer_iterations(1),
        solver.max_inner_iterations(1).max_outer_iterations(2),
    ] {
        let fit = problem.fit_with(&options).unwrap();
        assert_eq!(fit.termination(), Termination::IterationLimit);
        assert!(fit.outer_iterations() <= 2);
        check(&fit, &x, &y, &penalty, false);
        let gradient = 0.02 - 1.0 / (1.0 + fit.coefficients()[0].exp());
        close(fit.stopping_value(), gradient.abs(), 1e-12);
    }
}

#[test]
fn intercept_only_and_separated_models_stay_finite() {
    let x = Matrix::empty(4, 0);
    for y in [[0.0, 1.0, 1.0, 1.0], [0.0; 4], [1.0; 4]] {
        let penalty = Slope::new(vec![]);
        let fit = Problem::new(&x, Binomial::new(&y), penalty.clone())
            .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!(fit.intercept().is_finite());
        check(&fit, &x, &y, &penalty, true);
        if y[0] != y[1] {
            close(fit.intercept(), 3.0_f64.ln(), 1e-8);
        }
    }
    let x = Matrix::from_rows(&[&[-1000.0, -1000.0], &[1000.0, 1000.0]]);
    let y = [0.0, 1.0];
    let penalty = ChainFusedLasso::new(0.0, 0.0);
    let fit = Problem::new(&x, Binomial::new(&y).fit_intercept(false), penalty.clone())
        .normalize(Normalization::None)
        .fit_with(
            &Hybrid::<Vec<f64>>::new()
                .tolerance(1e-11)
                .max_outer_iterations(300)
                .max_iterations(30_000),
        )
        .unwrap();
    assert_eq!(
        fit.termination(),
        Termination::Converged,
        "outer={}, inner={}, mapping={}, eta={:?}",
        fit.outer_iterations(),
        fit.iterations(),
        fit.stopping_value(),
        fit.predict(&x).unwrap()
    );
    assert!(
        fit.predict(&x)
            .unwrap()
            .iter()
            .all(|eta| eta.is_finite() && eta.abs() > 30.0)
    );
    check(&fit, &x, &y, &penalty, false);
}

#[test]
fn invalid_options_and_data_fail_before_preparation() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [0.0, 1.0];
    let problem = Problem::new(&x, Binomial::new(&y), ClusteredLasso::new(0.1, 0.0));
    for solver in [
        Hybrid::<Vec<f64>>::new().max_outer_iterations(0),
        Hybrid::new().max_inner_iterations(0),
        Hybrid::new().tolerance(f64::NAN),
    ] {
        assert!(matches!(
            problem.fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
    for response in [&[0.0][..], &[0.0, f64::NAN], &[0.0, 2.0]] {
        assert!(
            Problem::new(&x, Binomial::new(response), ClusteredLasso::new(0.1, 0.0))
                .fit_with(&Hybrid::<Vec<f64>>::new())
                .is_err()
        );
    }
    assert_eq!(x.visits.get(), 0);
    assert!(matches!(
        problem.fit_with(
            &Hybrid::<Vec<f64>>::new().step_policy(HybridStep::Backtracking {
                initial_curvature: Some(1e-9),
                max_backtracks: 0
            })
        ),
        Err(FitError::LineSearchFailure { .. })
    ));
}

#[test]
fn damping_checks_original_objective_and_rebuilds_accepted_partitions() {
    let mut state = 7_u64;
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let rows: Vec<Vec<f64>> = (0..12)
        .map(|_| (0..4).map(|_| (random() - 0.5) * 10.0).collect())
        .collect();
    let y: Vec<_> = (0..12).map(|_| f64::from(random() > 0.5)).collect();
    let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let penalty = ClusteredLasso::new(1e-5, 1e-5);
    let problem =
        Problem::new(&x, Binomial::new(&y), penalty.clone()).normalize(Normalization::None);
    let mut previous = 2.0_f64.ln();
    let mut damped = false;
    for count in 1..=20 {
        let fit = problem
            .fit_with(
                &Hybrid::<Vec<f64>>::new()
                    .tolerance(1e-8)
                    .max_outer_iterations(count),
            )
            .unwrap();
        assert!(fit.objective() <= previous + 2e-14);
        previous = fit.objective();
        damped |= fit.outer_backtracks() > 0;
        check(&fit, &x, &y, &penalty, true);
        for (j, &label) in fit.cluster_labels().iter().enumerate() {
            assert_eq!(fit.coefficients()[j], fit.cluster_values()[label]);
        }
        if fit.termination() == Termination::Converged {
            break;
        }
    }
    assert!(damped);
    assert!(matches!(
        problem.fit_with(
            &Hybrid::<Vec<f64>>::new()
                .tolerance(1e-8)
                .max_outer_backtracks(0)
        ),
        Err(FitError::LineSearchFailure { .. })
    ));
}
