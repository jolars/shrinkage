//! Regularization thresholds, sequences, and path optimality.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    AxisPenalty, Binomial, CoefficientLayout, CoordinateDescent, ElasticNet, FitError, Gaussian,
    L1, Lasso, MultivariateGaussian, Normalization, PathOptions, PathPenalty, PenaltyAxis,
    PenaltyMaximum, Problem, ProximalGradient, Ridge, Slope, Termination,
};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-7, "{a} != {b}");
}

#[test]
fn lasso_path_retains_order_and_analytical_solutions() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-1.0, -7.0, 9.0, 3.0];
    let model = Lasso::new(99.0).tolerance(1e-10);
    let path = model.fit_path(&x, &y, &[5.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    assert_eq!(path.len(), 5);
    assert_eq!(path.lambdas(), Some([5.0, 1.0, 0.0, 1.0, 1.0].as_slice()));
    for (penalty, fit) in path.iter() {
        assert_eq!(fit.termination(), Termination::Converged);
        close(fit.intercept(), 1.0);
        close(fit.coefficients()[0], (5.0 - penalty.lambda()).max(0.0));
        close(fit.coefficients()[1], -(3.0 - penalty.lambda()).max(0.0));
        assert!(fit.kkt_violation() < 1e-9);
    }
    assert_eq!(path.fits()[4].iterations(), 0);
    assert!(model.fit_path(&x, &y, &[]).unwrap().is_empty());
}

#[test]
fn default_lasso_path_resolves_zero_penalty_criterion_per_point() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let path = Lasso::new(1.0)
        .fit_path(&x, &[0.0, 1.0, 1.0, 4.0], &[1.0, 0.0])
        .unwrap();
    assert_eq!(path.fits()[1].termination(), Termination::Converged);
    assert!(matches!(
        path.fits()[1].stopping_criterion(),
        shrinkage::StoppingCriterion::KktViolation { .. }
    ));
}

#[test]
fn automatic_lasso_uses_normalized_null_gradient_and_keeps_metadata() {
    let x = Matrix::from_rows(&[&[8.0], &[10.0], &[12.0]]);
    let y = [1.0, 3.0, 5.0];
    let problem = Problem::new(&x, Gaussian::new(&y), L1::new(99.0));
    let solver = CoordinateDescent::new().tolerance(1e-10);
    let maximum = problem.maximum_penalty_with(&solver).unwrap();
    let expected = (8.0_f64 / 3.0).sqrt();
    close(maximum.finite().unwrap(), expected);
    let path = problem
        .fit_regularization_path_with(
            &solver,
            &PathOptions::new().nlambda(3).lambda_min_ratio(0.01),
        )
        .unwrap();
    close(path.lambda_start().unwrap(), expected);
    close(path.lambdas().unwrap()[1], expected * 0.1);
    close(path.lambdas().unwrap()[2], expected * 0.01);
    assert_eq!(path.maximum_penalty(), Some(maximum));
    assert_eq!(path.fits()[0].iterations(), 0);
    assert_eq!(path.fits()[0].coefficients(), &[0.0]);
    close(path.fits()[2].predict(&x).unwrap()[0], 1.02);
}

#[test]
fn proximal_paths_preserve_independent_strengths_and_iteration_limits() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [-3.0, 3.0];
    let problem = Problem::new(&x, Gaussian::new(&y), L1::new(99.0));
    let solver = ProximalGradient::new().tolerance(1e-10);
    let path = problem
        .fit_path_with(
            &solver,
            vec![ElasticNet::new(1.0, 0.5), ElasticNet::new(0.5, 1.0)],
        )
        .unwrap();
    close(path.fits()[0].coefficients()[0], 2.0 / 1.5);
    close(path.fits()[1].coefficients()[0], 2.5 / 2.0);
    assert_eq!(path.penalties()[1].l2(), 1.0);
    let solver = solver.max_iterations(1).initial_step(0.1);
    let path = problem
        .fit_path_with(&solver, vec![Ridge::new(0.2), Ridge::new(0.2)])
        .unwrap();
    assert!(
        path.fits()
            .iter()
            .all(|fit| fit.termination() == Termination::IterationLimit)
    );
    assert!(path.fits()[1].objective() < path.fits()[0].objective());
}

#[test]
fn mixing_and_ridge_anchors_are_explicit() {
    let gradient = [3.0, -2.0];
    for alpha in [0.001, 0.5, 1.0] {
        let penalty = ElasticNet::from_alpha(0.0, alpha);
        close(penalty.alpha().unwrap(), alpha);
        assert_eq!(
            penalty.maximum_penalty(&gradient).unwrap(),
            PenaltyMaximum::Finite(3.0 / alpha)
        );
        let point = penalty.at_strength(2.0).unwrap();
        close(point.l1(), 2.0 * alpha);
        close(point.l2(), 2.0 * (1.0 - alpha));
    }
    assert_eq!(
        Ridge::new(1.0).maximum_penalty(&gradient).unwrap(),
        PenaltyMaximum::Unbounded
    );
    close(Ridge::new(1.0).path_anchor(&gradient).unwrap(), 3000.0);
    close(
        ElasticNet::from_alpha(1.0, 0.0)
            .path_anchor(&gradient)
            .unwrap(),
        3000.0,
    );
    assert!(
        ElasticNet::from_alpha(1.0, -0.1)
            .maximum_penalty(&gradient)
            .is_err()
    );
    assert!(
        ElasticNet::from_alpha(-1.0, 0.5)
            .maximum_penalty(&gradient)
            .is_err()
    );
}

#[test]
fn binomial_threshold_uses_optimized_intercept_and_rejects_undefined_null_model() {
    let x = Matrix::from_rows(&[&[-2.0], &[0.0], &[1.0], &[3.0]]);
    let y = [0.0, 0.0, 1.0, 0.0];
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(1.0)).normalize(Normalization::None);
    let solver = ProximalGradient::new().tolerance(1e-10);
    assert_eq!(
        problem.maximum_penalty_with(&solver).unwrap(),
        PenaltyMaximum::Finite(0.125)
    );
    let path = problem
        .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(1))
        .unwrap();
    assert_eq!(path.fits()[0].iterations(), 0);
    close(path.fits()[0].intercept(), (1.0_f64 / 3.0).ln());
    let constant = Problem::new(&x, Binomial::new(&[1.0; 4]), L1::new(1.0));
    assert!(matches!(
        constant.maximum_penalty_with(&solver),
        Err(FitError::InvalidInput { .. })
    ));
    let fixed = Problem::new(
        &x,
        Binomial::new(&[1.0; 4]).fit_intercept(false),
        L1::new(1.0),
    );
    assert!(fixed.maximum_penalty_with(&solver).is_ok());
    use shrinkage::NullModelDatafit;
    let tiny_response = [f64::from_bits(1), 0.0];
    let tiny = Binomial::new(&tiny_response);
    assert!(tiny.null_intercepts().unwrap()[0].is_finite());
}

#[test]
fn slope_and_axis_thresholds_use_cumulative_weights() {
    let slope = Slope::new(vec![2.0, 1.0]);
    close(
        slope
            .maximum_penalty(&[3.0, -3.0])
            .unwrap()
            .finite()
            .unwrap(),
        2.0,
    );
    let layout = CoefficientLayout::new(2, 2).unwrap();
    let gradient = [3.0, 3.0, 6.0, 0.0];
    for (axis, weights, expected) in [
        (PenaltyAxis::All, vec![2.0, 1.0, 1.0, 0.0], 3.0),
        (PenaltyAxis::Features, vec![2.0, 1.0], 3.0),
        (PenaltyAxis::Responses, vec![2.0, 1.0], 3.0),
    ] {
        let term = AxisPenalty::new(layout, axis, Slope::new(weights));
        close(
            term.maximum_penalty(&gradient).unwrap().finite().unwrap(),
            expected,
        );
    }
    assert!(Slope::new(vec![0.0, 0.0]).path_anchor(&[1.0, 0.0]).is_err());
}

#[test]
fn multivariate_path_retains_response_intercepts_and_layout() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [-1.0, 5.0, 4.0, 0.0];
    let layout = CoefficientLayout::new(1, 2).unwrap();
    let term = |lambda| AxisPenalty::new(layout, PenaltyAxis::Features, L1::new(lambda));
    let problem = Problem::new(&x, MultivariateGaussian::new(&y, 2, 2), term(99.0));
    let solver = ProximalGradient::new().tolerance(1e-10);
    let path = problem
        .fit_path_with(&solver, vec![term(3.0), term(1.0), term(0.0)])
        .unwrap();
    for (penalty, fit) in path.iter() {
        close(
            fit.coefficients()[0],
            (3.0 - penalty.penalty().lambda()).max(0.0),
        );
        close(
            fit.coefficients()[1],
            -(2.0 - penalty.penalty().lambda()).max(0.0),
        );
        assert_eq!(fit.intercepts(), &[2.0, 2.0]);
    }
    let auto = problem
        .fit_regularization_path_with(
            &solver,
            &PathOptions::new().nlambda(2).lambda_min_ratio(0.1),
        )
        .unwrap();
    close(auto.lambda_start().unwrap(), 3.0);
    assert_eq!(auto.fits()[0].iterations(), 0);
}

#[test]
fn sequence_options_validate_and_handle_zero_start() {
    let x = Matrix::from_rows(&[&[0.0], &[0.0]]);
    let model = Lasso::new(1.0);
    for options in [
        PathOptions::new().nlambda(0),
        PathOptions::new().lambda_min_ratio(0.0),
        PathOptions::new().lambda_min_ratio(1.1),
        PathOptions::new().lambda_start(f64::NAN),
    ] {
        assert!(
            model
                .fit_regularization_path(&x, &[2.0, 2.0], &options)
                .is_err()
        );
    }
    let path = model
        .fit_regularization_path(&x, &[2.0, 2.0], &PathOptions::new().nlambda(3))
        .unwrap();
    assert_eq!(path.lambdas(), Some([0.0; 3].as_slice()));
    assert!(path.fits().iter().all(|fit| fit.iterations() == 0));
    assert!(model.fit_path(&x, &[2.0, 2.0], &[1.0, -1.0]).is_err());
    let empty = Matrix::empty(2, 0);
    let path = model
        .fit_regularization_path(&empty, &[1.0, 3.0], &PathOptions::new().nlambda(2))
        .unwrap();
    assert!(path.fits().iter().all(|fit| fit.coefficients().is_empty()));
}

#[test]
fn automatic_slope_path_has_independent_orthogonal_solution() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-1.0, -7.0, 9.0, 3.0];
    let problem = Problem::new(&x, Gaussian::new(&y), Slope::new(vec![2.0, 1.0]));
    let path = problem
        .fit_regularization_path_with(
            &ProximalGradient::new().tolerance(1e-10),
            &PathOptions::new().nlambda(2).lambda_min_ratio(0.375),
        )
        .unwrap();
    close(path.lambda_start().unwrap(), 8.0 / 3.0);
    assert_eq!(path.fits()[0].coefficients(), &[0.0, 0.0]);
    close(path.fits()[1].coefficients()[0], 3.0);
    close(path.fits()[1].coefficients()[1], -2.0);
    assert_eq!(path.penalties()[1].weights(), &[2.0, 1.0]);
}

#[test]
fn automatic_mixing_paths_and_ridge_overrides_fit_the_actual_penalty() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [-3.0, 3.0];
    let solver = ProximalGradient::new().tolerance(1e-10);
    for alpha in [0.0, 0.25, 1.0] {
        let problem = Problem::new(&x, Gaussian::new(&y), ElasticNet::from_alpha(0.0, alpha));
        let path = problem
            .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(3))
            .unwrap();
        for (penalty, fit) in path.iter() {
            close(
                fit.coefficients()[0],
                (3.0 - penalty.l1()).max(0.0) / (1.0 + penalty.l2()),
            );
            assert_eq!(fit.termination(), Termination::Converged);
        }
        if alpha == 0.0 {
            assert_eq!(path.maximum_penalty(), Some(PenaltyMaximum::Unbounded));
            close(path.lambda_start().unwrap(), 3000.0);
        }
    }
    let ridge = Problem::new(&x, Gaussian::new(&y), Ridge::new(99.0));
    let path = ridge
        .fit_regularization_path_with(&solver, &PathOptions::new().nlambda(1).lambda_start(2.0))
        .unwrap();
    close(path.fits()[0].coefficients()[0], 1.0);
    assert_eq!(path.maximum_penalty(), Some(PenaltyMaximum::Unbounded));
    assert_eq!(path.lambda_start(), Some(2.0));
}

#[test]
fn correlated_paths_reintroduce_features_and_pass_full_kkt_checks() {
    let x = Matrix::from_rows(&[
        &[1.0, 1.1, -2.0, 0.0],
        &[2.0, 1.9, 1.0, 0.0],
        &[3.0, 3.2, 2.0, 0.0],
        &[4.0, 3.8, -1.0, 0.0],
        &[5.0, 5.1, 0.0, 0.0],
        &[6.0, 6.0, 3.0, 0.0],
    ]);
    let y = [3.0, -1.0, 2.0, 5.0, 1.0, 4.0];
    let problem = Problem::new(&x, Gaussian::new(&y), L1::new(1.0));
    let lambdas = [3.0, 0.1, 1.0, 0.01, 2.0, 0.1, 0.0];
    let penalties = || lambdas.iter().map(|&v| L1::new(v)).collect();
    let solver = CoordinateDescent::new()
        .tolerance(1e-9)
        .max_iterations(200_000);
    let screened = problem.fit_path_with(&solver, penalties()).unwrap();
    let complete = problem
        .fit_path_with(&solver.screening(false).working_sets(false), penalties())
        .unwrap();
    for ((penalty, fit), reference) in screened.iter().zip(complete.fits()) {
        assert_eq!(fit.termination(), Termination::Converged);
        close(fit.objective(), reference.objective());
        let predicted = fit.predict(&x).unwrap();
        let residual = y
            .iter()
            .zip(&predicted)
            .map(|(y, eta)| y - eta)
            .collect::<Vec<_>>();
        assert!(residual.iter().sum::<f64>().abs() < 1e-7);
        let centers = fit.preprocessing().centers().unwrap();
        let scales = fit.preprocessing().scales().unwrap();
        use shrinkage::lazymatrix::{RawColumn, RawColumns};
        for feature in 0..4 {
            let mut raw = 0.0;
            x.raw_column(feature)
                .for_each_stored(|row, value| raw += value * residual[row]);
            let c =
                (raw - centers[feature] * residual.iter().sum::<f64>()) / (6.0 * scales[feature]);
            let beta = fit.coefficients()[feature] * scales[feature];
            let violation = if beta == 0.0 {
                (c.abs() - penalty.lambda()).max(0.0)
            } else {
                (c - penalty.lambda() * beta.signum()).abs()
            };
            assert!(violation < 1e-7);
        }
    }
    assert!(screened.fits()[0].coefficients().iter().all(|v| *v == 0.0));
    assert!(screened.fits()[1].coefficients().iter().any(|v| *v != 0.0));
}

#[test]
fn dimension_defaults_and_numerical_failures_are_explicit() {
    for (x, ratio) in [
        (Matrix::from_rows(&[&[-1.0], &[1.0]]), 0.0001),
        (
            Matrix::from_rows(&[&[-1.0, -2.0, -3.0], &[1.0, 2.0, 3.0]]),
            0.01,
        ),
    ] {
        let path = Lasso::new(1.0)
            .fit_regularization_path(&x, &[-1.0, 1.0], &PathOptions::new().early_stopping(false))
            .unwrap();
        assert_eq!(path.len(), 100);
        close(
            path.lambdas().unwrap()[99] / path.lambda_start().unwrap(),
            ratio,
        );
    }
    assert!(matches!(
        ElasticNet::from_alpha(1.0, 1e-300).maximum_penalty(&[1e300]),
        Err(FitError::NumericalFailure { .. })
    ));
    assert!(L1::new(1.0).maximum_penalty(&[f64::NAN]).is_err());
}

#[test]
fn zero_feature_response_axis_slope_paths_keep_null_intercepts() {
    let x = Matrix::empty(2, 0);
    let y = [1.0, 3.0, -2.0, 4.0];
    let layout = CoefficientLayout::new(0, 2).unwrap();
    let penalty = AxisPenalty::new(layout, PenaltyAxis::Responses, Slope::new(vec![1.0, 0.5]));
    let problem = Problem::new(&x, MultivariateGaussian::new(&y, 2, 2), penalty);
    let path = problem
        .fit_regularization_path_with(&ProximalGradient::new(), &PathOptions::new().nlambda(2))
        .unwrap();
    assert_eq!(path.lambdas(), Some([0.0, 0.0].as_slice()));
    for fit in path.fits() {
        assert!(fit.coefficients().is_empty());
        assert_eq!(fit.intercepts(), &[2.0, 1.0]);
    }
}
