//! Coordinate IRLS fits and independent optimality checks.

#[path = "common/matrix.rs"]
mod matrix;
#[path = "irls/reference.rs"]
mod reference;

use matrix::Matrix;
use shrinkage::{
    Binomial, CoordinateDescent, CoordinatePenalty, Datafit, DiagonalHessianDatafit, ElasticNet,
    FitError, Gaussian, Irls, IrlsFit, L1, Normalization, PathOptions, Penalty, PenaltyMaximum,
    Problem, ProximalGradient, Ridge, SmoothDatafit, StoppingCriterion, Termination,
};

#[test]
fn binomial_lasso_solves_scalar_optimality_equation() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let fit = Problem::new(
        &x,
        Binomial::new(&[0.0, 1.0]).fit_intercept(false),
        L1::new(0.2),
    )
    .normalize(Normalization::None)
    .fit_with(&Irls::new().tolerance(1e-11))
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert!((fit.coefficients()[0] - 4.0_f64.ln()).abs() < 1e-9);
    assert!(fit.kkt_violation() <= 1e-11);
}

#[test]
fn intercept_recovers_empirical_log_odds() {
    let x = Matrix::empty(4, 0);
    let fit = Problem::new(&x, Binomial::new(&[0.0, 1.0, 1.0, 1.0]), Ridge::new(0.5))
        .fit_with(&Irls::new().tolerance(1e-11))
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert!((fit.intercept() - 3.0_f64.ln()).abs() < 1e-9);
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-7 * (1.0 + b.abs()), "{a} != {b}");
}

fn check_kkt(fit: &IrlsFit, rows: &[&[f64]], y: &[f64], intercept: bool, l1: f64, l2: f64) {
    let eta = fit.predict(&Matrix::from_rows(rows)).unwrap();
    let g: Vec<_> = eta
        .iter()
        .zip(y)
        .map(|(&eta, &y)| {
            let t = (-eta.abs()).exp();
            (if eta >= 0.0 {
                (1.0 - y) - t / (1.0 + t)
            } else {
                t / (1.0 + t) - y
            }) / rows.len() as f64
        })
        .collect();
    let mut violation = if intercept {
        g.iter().sum::<f64>().abs()
    } else {
        0.0
    };
    let mut regularization = 0.0;
    for (j, &beta) in fit.coefficients().iter().enumerate() {
        let center = fit.preprocessing().centers().map_or(0.0, |v| v[j]);
        let scale = fit.preprocessing().scales().map_or(1.0, |v| v[j]);
        let theta = beta * scale;
        let smooth = rows
            .iter()
            .zip(&g)
            .map(|(row, g)| (row[j] - center) / scale * g)
            .sum::<f64>()
            + l2 * theta;
        violation = violation.max(if theta == 0.0 {
            (smooth.abs() - l1).max(0.0)
        } else {
            (smooth + l1 * theta.signum()).abs()
        });
        regularization += l1 * theta.abs() + 0.5 * l2 * theta * theta;
    }
    close(fit.kkt_violation(), violation);
    close(
        fit.objective(),
        Binomial::new(y).value(&eta).unwrap() + regularization,
    );
    if fit.termination() == Termination::Converged {
        assert!(violation <= fit.stopping_threshold() * 1.01);
    }
}

#[test]
fn hessians_match_gradient_differences_and_preserve_tails() {
    for target in [0.0, 0.3, 1.0] {
        for eta in [-700.0, -40.0, -8.0, -0.2, 0.0, 0.2, 8.0, 40.0, 700.0] {
            let y = [target];
            let loss = Binomial::new(&y);
            let mut h = [0.0];
            loss.hessian_diagonal(&[eta], &mut h).unwrap();
            let mut plus = [0.0];
            let mut minus = [0.0];
            let d = 1e-4;
            loss.gradient(&[eta + d], &mut plus).unwrap();
            loss.gradient(&[eta - d], &mut minus).unwrap();
            assert!(h[0] > 0.0);
            if (eta > 0.0 && target == 1.0) || (eta < 0.0 && target == 0.0) || eta.abs() < 10.0 {
                assert!(((plus[0] - minus[0]) / (2.0 * d) - h[0]).abs() <= 1e-7 * h[0]);
            }
        }
    }
    let mut h = [0.0; 3];
    Gaussian::new(&[0.0; 3])
        .hessian_diagonal(&[0.0; 3], &mut h)
        .unwrap();
    assert_eq!(h, [1.0 / 3.0; 3]);
    assert!(
        Binomial::new(&[0.0])
            .hessian_diagonal(&[0.0], &mut [])
            .is_err()
    );
    assert!(
        Gaussian::new(&[0.0])
            .hessian_diagonal(&[f64::NAN], &mut [0.0])
            .is_err()
    );
}

#[test]
fn gaussian_agrees_with_existing_coordinate_descent() {
    let rows: &[&[f64]] = &[
        &[0.0, 1.0, 5.0, 0.0],
        &[1.0, 0.0, 5.0, 0.0],
        &[2.0, 3.0, 5.0, 0.0],
        &[0.0, -1.0, 5.0, 0.0],
        &[4.0, 0.0, 5.0, 0.0],
    ];
    let x = Matrix::from_rows(rows);
    let y = [1.0, 4.0, 2.0, 3.0, 8.0];
    for intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Auto,
            Normalization::Center,
            Normalization::Standardize,
        ] {
            let problem =
                Problem::new(&x, Gaussian::new(&y).fit_intercept(intercept), L1::new(0.1))
                    .normalize(normalization);
            let expected = problem
                .fit_with(&CoordinateDescent::new().tolerance(1e-10))
                .unwrap();
            let actual = problem.fit_with(&Irls::new().tolerance(1e-10)).unwrap();
            assert_eq!(actual.termination(), Termination::Converged);
            close(actual.objective(), expected.objective());
            for (a, b) in actual
                .predict(&x)
                .unwrap()
                .into_iter()
                .zip(expected.predict(&x).unwrap())
            {
                close(a, b);
            }
        }
    }
}

#[test]
fn paths_restore_features_and_match_full_sweeps_and_independent_fits() {
    let rows: &[&[f64]] = &[
        &[1.0, 1.1, -2.0, 0.0],
        &[2.0, 1.6, 1.0, 0.0],
        &[3.0, 3.8, 2.0, 0.0],
        &[4.0, 3.0, -1.0, 0.0],
        &[5.0, 6.0, 0.0, 0.0],
        &[6.0, 5.5, 3.0, 0.0],
    ];
    let x = Matrix::from_rows(rows);
    let y = [0.1, 0.2, 0.9, 0.8, 0.3, 0.9];
    let solver = Irls::new().tolerance(1e-10).max_inner_iterations(10_000);
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(1.0));
    let lambdas = [1.0, 0.01, 0.1, 0.0, 0.01, 2.0, 0.1];
    let penalties = || lambdas.into_iter().map(L1::new).collect();
    let path = problem.fit_path_with(&solver, penalties()).unwrap();
    let full = problem
        .fit_path_with(&solver.working_sets(false), penalties())
        .unwrap();
    for (((&lambda, fit), other), penalty) in lambdas
        .iter()
        .zip(path.fits())
        .zip(full.fits())
        .zip(lambdas.map(L1::new))
    {
        let independent = Problem::new(&x, Binomial::new(&y), penalty)
            .fit_with(&solver)
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        check_kkt(fit, rows, &y, true, lambda, 0.0);
        close(fit.objective(), other.objective());
        close(fit.objective(), independent.objective());
        for (a, b) in fit
            .predict(&x)
            .unwrap()
            .into_iter()
            .zip(independent.predict(&x).unwrap())
        {
            close(a, b);
        }
    }
    assert!(path.fits()[0].coefficients().iter().all(|&v| v == 0.0));
    assert!(path.fits()[1].coefficients().iter().any(|&v| v != 0.0));
    assert!(
        problem
            .fit_path_with(&solver, Vec::<L1>::new())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn automatic_paths_reuse_existing_threshold_and_anchor_conventions() {
    let x = Matrix::from_rows(&[&[-2.0], &[0.0], &[1.0], &[3.0]]);
    let y = [0.0, 0.0, 1.0, 0.0];
    let solver = Irls::new().tolerance(1e-10);
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(1.0)).normalize(Normalization::None);
    assert_eq!(
        problem.maximum_penalty_with(&solver).unwrap(),
        PenaltyMaximum::Finite(0.125)
    );
    let path = problem
        .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(4))
        .unwrap();
    assert_eq!(path.fits()[0].iterations(), 0);
    close(path.fits()[0].intercept(), (1.0_f64 / 3.0).ln());
    assert!(
        Problem::new(&x, Binomial::new(&[1.0; 4]), L1::new(0.1))
            .maximum_penalty_with(&solver)
            .is_err()
    );
    assert!(
        Problem::new(
            &x,
            Binomial::new(&[1.0; 4]).fit_intercept(false),
            L1::new(0.1)
        )
        .maximum_penalty_with(&solver)
        .is_ok()
    );
    let ridge = Problem::new(&x, Binomial::new(&y), Ridge::new(1.0));
    assert_eq!(
        ridge.maximum_penalty_with(&solver).unwrap(),
        PenaltyMaximum::Unbounded
    );
    assert!(
        ridge
            .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(2))
            .is_ok()
    );
    let elastic = Problem::new(&x, Binomial::new(&y), ElasticNet::from_alpha(0.0, 0.5));
    let path = elastic
        .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(3))
        .unwrap();
    assert!(path.penalties().iter().all(|p| p.alpha() == Some(0.5)));
}

#[test]
fn normalization_back_transforms_predictions_and_penalty_scale() {
    let rows: &[&[f64]] = &[
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[2.0, 3.0],
        &[0.0, -1.0],
        &[4.0, 0.0],
    ];
    let y = [0.0, 0.3, 1.0, 0.0, 1.0];
    let x = Matrix::from_rows(rows);
    let centers = [0.5, -1.0];
    let scales = [2.0, 0.5];
    let transformed: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| (0..2).map(|j| (row[j] - centers[j]) / scales[j]).collect())
        .collect();
    let explicit = Matrix::from_rows(&transformed.iter().map(Vec::as_slice).collect::<Vec<_>>());
    for intercept in [false, true] {
        let solver = Irls::new().tolerance(1e-10);
        let penalty = ElasticNet::new(0.05, 0.1);
        let fit = Problem::new(&x, Binomial::new(&y).fit_intercept(intercept), penalty)
            .with_centers(centers.to_vec())
            .with_scales(scales.to_vec())
            .fit_with(&solver)
            .unwrap();
        let other = Problem::new(
            &explicit,
            Binomial::new(&y).fit_intercept(intercept),
            penalty,
        )
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
        for (a, b) in fit
            .predict(&x)
            .unwrap()
            .into_iter()
            .zip(other.predict(&explicit).unwrap())
        {
            close(a, b);
        }
        check_kkt(&fit, rows, &y, intercept, 0.05, 0.1);
        assert!(fit.predict(&Matrix::empty(0, 2)).unwrap().is_empty());
        assert!(fit.predict(&Matrix::empty(2, 3)).is_err());
    }
}

#[test]
fn validates_inputs_and_stopping_rules_and_returns_fresh_limit_diagnostics() {
    let rows: &[&[f64]] = &[&[-2.0], &[1.0], &[3.0]];
    let x = Matrix::from_rows(rows);
    let y = [0.0, 0.2, 1.0];
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(0.001));
    for solver in [
        Irls::new().tolerance(0.0),
        Irls::new().tolerance(f64::NAN),
        Irls::new().max_iterations(0),
        Irls::new().max_inner_iterations(0),
    ] {
        assert!(matches!(
            problem.fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
    for rule in [
        StoppingCriterion::duality_gap(1e-6),
        StoppingCriterion::proximal_gradient_mapping(1e-6),
    ] {
        assert!(matches!(
            problem.fit_with(&Irls::new().terminate_on(rule)),
            Err(FitError::UnsupportedCombination { .. })
        ));
    }
    let fit = problem
        .fit_with(
            &Irls::new()
                .max_iterations(1)
                .max_inner_iterations(1)
                .tolerance(1e-12),
        )
        .unwrap();
    assert_eq!(fit.termination(), Termination::IterationLimit);
    assert_eq!(fit.iterations(), 1);
    assert_eq!(fit.coordinate_sweeps(), 1);
    check_kkt(&fit, rows, &y, true, 0.001, 0.0);
    for y in [
        vec![],
        vec![-1.0; 3],
        vec![f64::NAN; 3],
        vec![f64::INFINITY; 3],
        vec![1.1; 3],
    ] {
        assert!(
            Problem::new(&x, Binomial::new(&y), L1::new(0.1))
                .fit_with(&Irls::new())
                .is_err()
        );
    }
    assert!(
        Problem::new(&Matrix::empty(0, 1), Binomial::new(&[]), L1::new(0.1))
            .fit_with(&Irls::new())
            .is_err()
    );
    assert!(
        Problem::new(&x, Binomial::new(&y), L1::new(-1.0))
            .fit_with(&Irls::new())
            .is_err()
    );
}

#[test]
fn separated_and_constant_boundary_models_remain_finite() {
    for y in [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]] {
        let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
        let fit = Problem::new(&x, Binomial::new(&y), L1::new(0.0))
            .fit_with(&Irls::new().tolerance(1e-10))
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!(fit.objective().is_finite());
        assert!(fit.intercept().is_finite());
        assert!(fit.coefficients().iter().all(|v| v.is_finite()));
    }
}

#[test]
fn ridge_and_elastic_net_match_proximal_gradient() {
    let x = Matrix::from_rows(&[&[-2.0, 1.0], &[-1.0, 0.0], &[1.0, 1.0], &[2.0, -1.0]]);
    let y = [0.0, 0.1, 0.7, 1.0];
    for (l1, l2) in [(0.0, 0.1), (0.05, 0.1)] {
        let problem = Problem::new(&x, Binomial::new(&y), ElasticNet::new(l1, l2));
        let fit = problem.fit_with(&Irls::new().tolerance(1e-10)).unwrap();
        let other = problem
            .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-10))
            .unwrap();
        close(fit.objective(), other.objective());
        for (a, b) in fit.coefficients().iter().zip(other.coefficients()) {
            close(*a, *b);
        }
    }
}

struct Exponential;

impl Datafit for Exponential {
    type Error = FitError;
    fn value(&self, eta: &[f64]) -> Result<f64, FitError> {
        let value = eta[0].exp() - 100.0 * eta[0];
        if value.is_finite() {
            Ok(value)
        } else {
            Err(FitError::NumericalFailure {
                operation: "evaluating an exponential loss",
                iteration: 0,
            })
        }
    }
}
impl SmoothDatafit for Exponential {
    fn nobs(&self) -> usize {
        1
    }
    fn fits_intercept(&self) -> bool {
        true
    }
    fn gradient(&self, eta: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        output[0] = eta[0].exp() - 100.0;
        Ok(())
    }
}
impl DiagonalHessianDatafit for Exponential {
    fn hessian_diagonal(&self, eta: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        output[0] = eta[0].exp();
        Ok(())
    }
}

#[test]
fn external_datafit_backtracks_against_the_original_loss() {
    let x = Matrix::empty(1, 0);
    let problem = Problem::new(&x, Exponential, Ridge::new(0.0));
    assert!(matches!(
        problem.fit_with(&Irls::new().max_backtracks(0)),
        Err(FitError::LineSearchFailure { iteration: 1, .. })
    ));
    let fit = problem.fit_with(&Irls::new().tolerance(1e-9)).unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.intercept(), 100.0_f64.ln());
    assert!(fit.objective() < 1.0);
}

struct IndexedL1([f64; 2]);
impl Penalty for IndexedL1 {
    type Error = FitError;
    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        assert_eq!(coefficients.len(), 2);
        Ok(coefficients
            .iter()
            .zip(self.0)
            .map(|(v, w)| v.abs() * w)
            .sum())
    }
}
impl CoordinatePenalty for IndexedL1 {
    fn minimize_coordinate(&self, j: usize, linear: f64, curvature: f64) -> Result<f64, FitError> {
        L1::new(self.0[j]).minimize_coordinate(j, linear, curvature)
    }
    fn coordinate_violation(&self, j: usize, value: f64, gradient: f64) -> Result<f64, FitError> {
        L1::new(self.0[j]).coordinate_violation(j, value, gradient)
    }
}

#[test]
fn external_coordinate_penalty_needs_no_proximal_map() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let fit = Problem::new(
        &x,
        Gaussian::new(&[3.0, 3.0]).fit_intercept(false),
        IndexedL1([0.5, 1.0]),
    )
    .normalize(Normalization::None)
    .fit_with(&Irls::new().tolerance(1e-10))
    .unwrap();
    close(fit.coefficients()[0], 2.0);
    close(fit.coefficients()[1], 1.0);
}

struct ShiftedRidge;
impl Penalty for ShiftedRidge {
    type Error = FitError;
    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        Ok(coefficients.iter().map(|v| 0.5 * (v - 1.0).powi(2)).sum())
    }
}
impl CoordinatePenalty for ShiftedRidge {
    fn minimize_coordinate(&self, _: usize, linear: f64, curvature: f64) -> Result<f64, FitError> {
        Ok((linear + 1.0) / (curvature + 1.0))
    }
    fn coordinate_violation(&self, _: usize, value: f64, gradient: f64) -> Result<f64, FitError> {
        Ok((gradient + value - 1.0).abs())
    }
}

#[test]
fn zero_design_columns_still_minimize_external_penalties() {
    let x = Matrix::empty(2, 1);
    let fit = Problem::new(
        &x,
        Gaussian::new(&[0.0; 2]).fit_intercept(false),
        ShiftedRidge,
    )
    .fit_with(&Irls::new())
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(fit.coefficients(), &[1.0]);
}

#[test]
fn unpenalized_constant_columns_work_with_full_sweeps() {
    let x = Matrix::from_rows(&[&[3.0], &[3.0], &[3.0], &[3.0], &[3.0]]);
    let fit = Problem::new(&x, Gaussian::new(&[1.0, 2.0, 4.0, 0.1, -3.2]), L1::new(0.0))
        .fit_with(&Irls::new().working_sets(false))
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.intercept(), 0.78);
    assert_eq!(fit.coefficients(), &[0.0]);
}

struct InvalidHessian(f64);
impl Datafit for InvalidHessian {
    type Error = FitError;
    fn value(&self, eta: &[f64]) -> Result<f64, FitError> {
        Gaussian::new(&[1.0, 2.0]).value(eta)
    }
}
impl SmoothDatafit for InvalidHessian {
    fn nobs(&self) -> usize {
        2
    }
    fn fits_intercept(&self) -> bool {
        false
    }
    fn gradient(&self, eta: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        Gaussian::new(&[1.0, 2.0]).gradient(eta, output)
    }
}
impl DiagonalHessianDatafit for InvalidHessian {
    fn hessian_diagonal(&self, _: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        output.fill(self.0);
        Ok(())
    }
}

#[test]
fn invalid_external_hessians_are_rejected_before_stabilization() {
    let x = Matrix::from_rows(&[&[1.0], &[2.0]]);
    for h in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            Problem::new(&x, InvalidHessian(h), L1::new(0.0)).fit_with(&Irls::new()),
            Err(FitError::InvalidInput { .. })
        ));
    }
}

#[test]
fn paths_retain_iteration_limit_results_and_warm_start_repeated_penalties() {
    let x = Matrix::from_rows(&[&[-2.0], &[-1.0], &[1.0], &[2.0]]);
    let y = [0.0, 0.1, 0.7, 1.0];
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(0.01));
    let path = problem
        .fit_path_with(
            &Irls::new().max_iterations(1).tolerance(1e-12),
            vec![L1::new(0.01); 3],
        )
        .unwrap();
    assert_eq!(path.len(), 3);
    assert!(
        path.fits()
            .iter()
            .all(|f| f.termination() == Termination::IterationLimit)
    );
    assert!(
        path.fits()
            .windows(2)
            .all(|pair| pair[1].objective() < pair[0].objective())
    );
    let path = problem
        .fit_path_with(&Irls::new().tolerance(1e-10), vec![L1::new(0.01); 2])
        .unwrap();
    assert_eq!(path.fits()[1].iterations(), 0);
    assert_eq!(path.fits()[1].coordinate_sweeps(), 0);
}
