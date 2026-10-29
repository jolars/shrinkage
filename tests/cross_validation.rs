//! Fold validation, training-only paths, selection, and full-data refitting.

#[path = "common/cv_matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, CrossValidation, CrossValidationError,
    FoldAssignments, Gaussian, Hybrid, L1, Lasso, Normalization, PathOptions, Problem, RowSubset,
    SelectionRule, Slope, Termination, ValidationMetric,
};

#[test]
fn folds_cover_every_row_and_reproduce_seeded_assignments() {
    let folds = FoldAssignments::kfold(23, 5, 42).unwrap();
    assert_eq!(folds.assignments().len(), 23);
    assert_eq!(folds.nfolds(), 5);
    assert_eq!(folds, FoldAssignments::kfold(23, 5, 42).unwrap());
    assert_ne!(folds, FoldAssignments::kfold(23, 5, 43).unwrap());
    let sizes = folds.fold_sizes();
    assert_eq!(sizes.iter().sum::<usize>(), 23);
    assert_eq!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap(), 1);
    for assignments in [vec![], vec![0], vec![0, 0], vec![0, 2], vec![usize::MAX, 0]] {
        assert!(FoldAssignments::new(assignments).is_err());
    }
    for (n, k) in [(0, 2), (3, 1), (3, 4)] {
        assert!(FoldAssignments::kfold(n, k, 0).is_err());
    }
}

#[test]
fn gaussian_paths_match_independent_training_fits_and_refit_selection() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[20.0]]);
    let y = [1.0, 2.0, 5.0, 6.0, 9.0, -1.0];
    let folds = FoldAssignments::new(vec![0, 1, 2, 0, 1, 2]).unwrap();
    let cv = CrossValidation::new(folds).selection_rule(SelectionRule::OneStandardError);
    let solver = shrinkage::CoordinateDescent::new().tolerance(1e-10);
    let penalties = vec![L1::new(5.0), L1::new(0.3), L1::new(0.05)];
    let problem = Problem::new(&x, Gaussian::new(&y), L1::new(1.0));
    let result = problem
        .cross_validate_path_with(&solver, penalties, &cv)
        .unwrap();
    assert_eq!(result.metric(), ValidationMetric::MeanSquaredError);
    assert_eq!(result.refit_path().len(), 3);
    assert!(result.relative_strengths().is_none());
    for fold in result.folds() {
        let training_rows = (0..y.len())
            .filter(|i| !fold.validation_rows().contains(i))
            .collect::<Vec<_>>();
        let train_x = Matrix::from_rows(
            &training_rows
                .iter()
                .map(|&i| [if i == 5 { 20.0 } else { i as f64 }])
                .collect::<Vec<_>>()
                .iter()
                .map(|r| r.as_slice())
                .collect::<Vec<_>>(),
        );
        let train_y = training_rows.iter().map(|&i| y[i]).collect::<Vec<_>>();
        for (j, (penalty, fit)) in fold.path().iter().enumerate() {
            let cold = Problem::new(&train_x, Gaussian::new(&train_y), *penalty)
                .fit_with(&solver)
                .unwrap();
            close(fit.intercept(), cold.intercept());
            close(fit.coefficients()[0], cold.coefficients()[0]);
            close(
                fit.preprocessing().centers().unwrap()[0],
                training_rows
                    .iter()
                    .map(|&i| if i == 5 { 20.0 } else { i as f64 })
                    .sum::<f64>()
                    / training_rows.len() as f64,
            );
            let predictions = cold.predict(&x).unwrap();
            let expected = fold
                .validation_rows()
                .iter()
                .map(|&i| (y[i] - predictions[i]).powi(2))
                .sum::<f64>()
                / fold.validation_rows().len() as f64;
            close(fold.scores()[j], expected);
        }
    }
    for j in 0..3 {
        let scores = result
            .folds()
            .iter()
            .map(|f| f.scores()[j])
            .collect::<Vec<_>>();
        let mean = scores.iter().sum::<f64>() / 3.0;
        let se = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / 6.0).sqrt();
        close(result.mean_scores()[j], mean);
        close(result.standard_errors()[j], se);
    }
    let min = (0..3)
        .min_by(|&a, &b| result.mean_scores()[a].total_cmp(&result.mean_scores()[b]))
        .unwrap();
    let selected = (0..3)
        .find(|&i| {
            result.mean_scores()[i] <= result.mean_scores()[min] + result.standard_errors()[min]
        })
        .unwrap();
    assert_eq!(result.minimum_index(), min);
    assert_eq!(result.selected_index(), selected);
    let cold = Problem::new(&x, Gaussian::new(&y), *result.penalty())
        .fit_with(&solver)
        .unwrap();
    close(result.fit().intercept(), cold.intercept());
    close(result.fit().coefficients()[0], cold.coefficients()[0]);

    let automatic = problem
        .cross_validate_regularization_path_with(
            &solver,
            &PathOptions::new().nlambda(4).lambda_min_ratio(0.1),
            &cv,
        )
        .unwrap();
    assert_eq!(automatic.relative_strengths().unwrap().len(), 4);
    for fold in automatic.folds() {
        assert_eq!(fold.path().len(), 4);
        for (&lambda, &fraction) in fold
            .path()
            .lambdas()
            .unwrap()
            .iter()
            .zip(automatic.relative_strengths().unwrap())
        {
            close(lambda, fold.path().lambda_start().unwrap() * fraction);
        }
    }
}

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-6 * (1.0 + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

#[test]
fn stratification_balances_classes_and_total_fold_sizes() {
    let y = [vec![0.0; 17], vec![1.0; 8]].concat();
    let folds = FoldAssignments::stratified_binomial(&y, 4, 7).unwrap();
    assert_eq!(
        folds,
        FoldAssignments::stratified_binomial(&y, 4, 7).unwrap()
    );
    assert_ne!(
        folds,
        FoldAssignments::stratified_binomial(&y, 4, 8).unwrap()
    );
    for class in [0.0, 1.0] {
        let counts = (0..4)
            .map(|fold| {
                y.iter()
                    .zip(folds.assignments())
                    .filter(|(target, assignment)| **target == class && **assignment == fold)
                    .count()
            })
            .collect::<Vec<_>>();
        assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1);
    }
    assert!(
        folds.fold_sizes().iter().max().unwrap() - folds.fold_sizes().iter().min().unwrap() <= 1
    );
    for y in [
        &[0.0, 0.5, 1.0][..],
        &[0.0, f64::NAN, 1.0],
        &[0.0, 2.0, 1.0],
    ] {
        assert!(FoldAssignments::stratified_binomial(y, 2, 0).is_err());
    }
}

#[test]
fn metrics_use_mean_losses_and_stable_logits() {
    close(
        ValidationMetric::MeanSquaredError
            .score(&[1.0, 5.0], &[2.0, 2.0])
            .unwrap(),
        5.0,
    );
    close(
        ValidationMetric::MeanAbsoluteError
            .score(&[1.0, 5.0], &[2.0, 2.0])
            .unwrap(),
        2.0,
    );
    close(
        ValidationMetric::LogLoss
            .score(&[0.0, 1.0], &[-1000.0, 1000.0])
            .unwrap(),
        0.0,
    );
    close(
        ValidationMetric::LogLoss
            .score(&[0.0, 1.0], &[1000.0, -1000.0])
            .unwrap(),
        1000.0,
    );
    let y = [0.25, 0.75];
    let eta = [(1.0_f64 / 3.0).ln(), 3.0_f64.ln()];
    close(
        ValidationMetric::BinomialDeviance.score(&y, &eta).unwrap(),
        0.0,
    );
    close(
        ValidationMetric::LogLoss.score(&y, &eta).unwrap(),
        -0.25 * 0.25_f64.ln() - 0.75 * 0.75_f64.ln(),
    );
    for metric in [
        ValidationMetric::MeanSquaredError,
        ValidationMetric::MeanAbsoluteError,
        ValidationMetric::LogLoss,
        ValidationMetric::BinomialDeviance,
    ] {
        assert!(metric.score(&[], &[]).is_err());
        assert!(metric.score(&[0.0], &[0.0, 1.0]).is_err());
        assert!(metric.score(&[0.0], &[f64::NAN]).is_err());
        assert!(metric.score(&[f64::NAN], &[0.0]).is_err());
    }
    assert!(ValidationMetric::LogLoss.score(&[-0.1], &[0.0]).is_err());
}

#[test]
fn automatic_grid_and_preprocessing_ignore_held_out_data() {
    let rows = [
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 2.0],
        [3.0, 0.0],
        [4.0, 1.0],
        [5.0, 0.0],
    ];
    let mut changed = rows;
    changed[0] = [10000.0, -70000.0];
    changed[3] = [-9000.0, 300.0];
    let x = Matrix::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>());
    let changed_x = Matrix::from_rows(&changed.iter().map(|r| r.as_slice()).collect::<Vec<_>>());
    let y = [0.0, 2.0, 1.0, 4.0, 3.0, 7.0];
    let changed_y = [500.0, 2.0, 1.0, -400.0, 3.0, 7.0];
    let cv = CrossValidation::new(FoldAssignments::new(vec![0, 1, 2, 0, 1, 2]).unwrap());
    let options = PathOptions::new().nlambda(4).lambda_min_ratio(0.1);
    let model = Lasso::new(1.0).tolerance(1e-8);
    let original = model
        .cross_validate_regularization_path(&x, &y, &options, &cv)
        .unwrap();
    let changed = model
        .cross_validate_regularization_path(&changed_x, &changed_y, &options, &cv)
        .unwrap();
    let a = original.folds()[0].path();
    let b = changed.folds()[0].path();
    assert_eq!(a.lambdas(), b.lambdas());
    assert_eq!(a.lambda_start(), b.lambda_start());
    for (a, b) in a.fits().iter().zip(b.fits()) {
        assert_eq!(a.coefficients(), b.coefficients());
        assert_eq!(a.intercept(), b.intercept());
        assert_eq!(a.preprocessing().centers(), b.preprocessing().centers());
        assert_eq!(a.preprocessing().scales(), b.preprocessing().scales());
    }
    assert_ne!(original.folds()[0].scores(), changed.folds()[0].scores());
    let indices = [1, 2, 4, 5];
    let train_x = x.select_rows(&indices).unwrap();
    let train_y = indices.map(|i| y[i]);
    let cold = model
        .fit_regularization_path(&train_x, &train_y, &options.early_stopping(false))
        .unwrap();
    assert_eq!(a.lambdas(), cold.lambdas());
    let expected = model
        .fit_regularization_path(&x, &y, &options.early_stopping(false))
        .unwrap();
    assert_eq!(original.refit_path().lambdas(), expected.lambdas());
    assert_ne!(a.lambda_start(), expected.lambda_start());
}

#[test]
fn errors_identify_invalid_inputs_degenerate_training_and_nonconvergence() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let folds = FoldAssignments::new(vec![0, 0, 1, 1]).unwrap();
    let cv = CrossValidation::new(folds);
    let solver = Hybrid::<Vec<f64>>::new();
    let problem = Problem::new(
        &x,
        Binomial::new(&[0.0, 0.0, 1.0, 1.0]),
        Slope::new(vec![1.0]),
    );
    assert!(matches!(
        problem.cross_validate_path_with(&solver, vec![Slope::new(vec![0.1])], &cv),
        Err(CrossValidationError::Fold { fold: 0, .. })
    ));
    assert!(matches!(
        problem.cross_validate_regularization_path_with(&solver, &PathOptions::new(), &cv),
        Err(CrossValidationError::Fold { fold: 0, .. })
    ));
    let fixed = Problem::new(
        &x,
        Binomial::new(&[0.0, 0.0, 1.0, 1.0]).fit_intercept(false),
        Slope::new(vec![1.0]),
    );
    assert!(
        fixed
            .cross_validate_path_with(&solver, vec![Slope::new(vec![0.1])], &cv)
            .is_ok()
    );
    let model = Lasso::new(1.0);
    let y = [1.0, 2.0, 2.0, 4.0];
    assert!(model.cross_validate_path(&x, &y, &[], &cv).is_err());
    assert!(model.cross_validate_path(&x, &y, &[-1.0], &cv).is_err());
    assert!(model.cross_validate_path(&x, &y[..3], &[0.1], &cv).is_err());
    assert!(
        model
            .cross_validate_path(
                &x,
                &y,
                &[0.1],
                &cv.clone().metric(ValidationMetric::LogLoss)
            )
            .is_err()
    );
    assert!(
        model
            .clone()
            .with_centers(vec![0.0])
            .cross_validate_path(&x, &y, &[0.1], &cv)
            .is_err()
    );
    assert!(
        model
            .clone()
            .with_scales(vec![1.0])
            .cross_validate_path(&x, &y, &[0.1], &cv)
            .is_err()
    );
    assert!(
        model
            .cross_validate_regularization_path(&x, &y, &PathOptions::new().nlambda(0), &cv)
            .is_err()
    );
    let mismatched = CrossValidation::new(FoldAssignments::new(vec![0, 1, 0]).unwrap());
    assert!(
        model
            .cross_validate_path(&x, &y, &[0.1], &mismatched)
            .is_err()
    );
    let wrong_penalty = Problem::new(&x, Gaussian::new(&y), Slope::new(vec![1.0]));
    assert!(matches!(
        wrong_penalty.cross_validate_path_with(&solver, vec![Slope::new(vec![0.1, 0.05])], &cv),
        Err(CrossValidationError::Fold { fold: 0, .. })
    ));
    let zero = Matrix::empty(4, 0);
    let ties = model
        .cross_validate_path(&zero, &y, &[5.0, 1.0, 0.1], &cv)
        .unwrap();
    assert_eq!(ties.minimum_index(), 0);
    assert_eq!(ties.one_standard_error_index(), 0);
    let limited = Problem::new(&x, Gaussian::new(&y), L1::new(0.1)).cross_validate_path_with(
        &shrinkage::ProximalGradient::<Vec<f64>>::new()
            .max_iterations(1)
            .initial_step(0.001)
            .tolerance(1e-15),
        vec![L1::new(0.1)],
        &cv,
    );
    assert!(matches!(
        limited,
        Err(CrossValidationError::NonConverged {
            fold: Some(0),
            candidate: 0
        })
    ));
}

#[test]
fn every_hybrid_penalty_and_family_matches_independent_fits() {
    let rows = [
        [-2.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, -1.0, 0.0],
        [2.0, 0.0, 1.0],
        [3.0, 1.0, 0.0],
        [4.0, 0.0, -1.0],
        [0.0, 2.0, 0.0],
    ];
    let x = Matrix::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>());
    let y = [-1.0, 0.0, 1.0, 3.0, 2.0, 4.0, 5.0, 1.0];
    let binary = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let cv = CrossValidation::new(FoldAssignments::stratified_binomial(&binary, 3, 42).unwrap());
    let solver = Hybrid::<Vec<f64>>::new()
        .tolerance(1e-8)
        .max_iterations(100_000);
    let options = PathOptions::new().nlambda(3).lambda_min_ratio(0.2);
    macro_rules! check {
        ($family:ident, $y:expr, $penalty:expr, $candidates:expr) => {{
            let response = $y;
            let problem =
                Problem::new(&x, $family::new(response), $penalty).normalize(Normalization::Auto);
            for rule in [SelectionRule::MinimumError, SelectionRule::OneStandardError] {
                let cv = cv.clone().selection_rule(rule);
                let result = problem
                    .cross_validate_path_with(&solver, $candidates, &cv)
                    .unwrap();
                let repeated = problem
                    .cross_validate_path_with(&solver, $candidates, &cv)
                    .unwrap();
                assert_eq!(result.mean_scores(), repeated.mean_scores());
                assert_eq!(result.selected_index(), repeated.selected_index());
                for fold in result.folds() {
                    let train = (0..response.len())
                        .filter(|i| !fold.validation_rows().contains(i))
                        .collect::<Vec<_>>();
                    let train_x = x.select_rows(&train).unwrap();
                    let train_y = train.iter().map(|&i| response[i]).collect::<Vec<_>>();
                    for (j, (penalty, fit)) in fold.path().iter().enumerate() {
                        let cold = Problem::new(&train_x, $family::new(&train_y), penalty.clone())
                            .fit_with(&solver)
                            .unwrap();
                        assert_eq!(fit.termination(), Termination::Converged);
                        close(fit.objective(), cold.objective());
                        let held = x.select_rows(fold.validation_rows()).unwrap();
                        let predictions = fit.predict(&held).unwrap();
                        for (&a, b) in predictions.iter().zip(cold.predict(&held).unwrap()) {
                            close(a, b);
                        }
                        let held_y = fold
                            .validation_rows()
                            .iter()
                            .map(|&i| response[i])
                            .collect::<Vec<_>>();
                        close(
                            fold.scores()[j],
                            result.metric().score(&held_y, &predictions).unwrap(),
                        );
                    }
                }
                let cold = Problem::new(&x, $family::new(response), result.penalty().clone())
                    .fit_with(&solver)
                    .unwrap();
                for (&a, b) in result
                    .fit()
                    .predict(&x)
                    .unwrap()
                    .iter()
                    .zip(cold.predict(&x).unwrap())
                {
                    close(a, b);
                }
                let min = result.minimum_index();
                let expected = match rule {
                    SelectionRule::MinimumError => min,
                    SelectionRule::OneStandardError => (0..result.mean_scores().len())
                        .find(|&j| {
                            result.mean_scores()[j]
                                <= result.mean_scores()[min] + result.standard_errors()[min]
                        })
                        .unwrap(),
                };
                assert_eq!(result.selected_index(), expected);
            }
            let auto = problem
                .cross_validate_regularization_path_with(&solver, &options, &cv)
                .unwrap();
            for fold in auto.folds() {
                let train = (0..response.len())
                    .filter(|i| !fold.validation_rows().contains(i))
                    .collect::<Vec<_>>();
                let train_x = x.select_rows(&train).unwrap();
                let train_y = train.iter().map(|&i| response[i]).collect::<Vec<_>>();
                let independent = Problem::new(&train_x, $family::new(&train_y), $penalty)
                    .fit_regularization_path_with(&solver, &options.early_stopping(false))
                    .unwrap();
                assert_eq!(fold.path().lambdas(), independent.lambdas());
            }
        }};
    }
    macro_rules! family {
        ($family:ident, $y:expr) => {
            check!(
                $family,
                $y,
                ClusteredLasso::new(1.0, 0.2),
                vec![
                    ClusteredLasso::new(0.5, 0.1),
                    ClusteredLasso::new(0.1, 0.2),
                    ClusteredLasso::new(0.1, 0.02)
                ]
            );
            check!(
                $family,
                $y,
                ChainFusedLasso::new(1.0, 0.2).with_weights(vec![0.0, 2.0]),
                vec![
                    ChainFusedLasso::new(0.5, 0.1).with_weights(vec![0.0, 2.0]),
                    ChainFusedLasso::new(0.1, 0.2).with_weights(vec![1.0, 0.5]),
                    ChainFusedLasso::new(0.1, 0.02).with_weights(vec![0.5, 2.0])
                ]
            );
            check!(
                $family,
                $y,
                Slope::new(vec![1.0, 0.6, 0.2]),
                vec![
                    Slope::new(vec![0.5, 0.3, 0.1]),
                    Slope::new(vec![0.2, 0.05, 0.0]),
                    Slope::new(vec![0.1, 0.06, 0.02])
                ]
            );
        };
    }
    family!(Gaussian, &y);
    family!(Binomial, &binary);
}

#[test]
fn automatic_paths_keep_zero_anchors_single_points_and_common_default_ratios() {
    let x = Matrix::empty(7, 4);
    let y = [0.0, 1.0, 2.0, 4.0, 7.0, 6.0, 3.0];
    let cv = CrossValidation::new(FoldAssignments::new(vec![0, 0, 0, 0, 1, 1, 2]).unwrap());
    let model = Lasso::new(1.0);
    let options = PathOptions::new()
        .nlambda(6)
        .min_points(1)
        .deviance_change_tolerance(10.0);
    let result = model
        .cross_validate_regularization_path(&x, &y, &options, &cv)
        .unwrap();
    assert_eq!(result.relative_strengths().unwrap().last(), Some(&0.01));
    for fold in result.folds() {
        assert_eq!(fold.path().len(), 6);
        assert_eq!(fold.path().lambda_start(), Some(0.0));
        assert_eq!(fold.path().lambdas(), Some([0.0; 6].as_slice()));
    }
    assert_eq!(result.minimum_index(), 0);
    let single = model
        .cross_validate_regularization_path(&x, &y, &PathOptions::new().nlambda(1), &cv)
        .unwrap();
    assert_eq!(single.relative_strengths(), Some([1.0].as_slice()));
    let fixed = model
        .cross_validate_regularization_path(
            &x,
            &y,
            &PathOptions::new()
                .nlambda(2)
                .lambda_start(3.0)
                .lambda_min_ratio(0.1),
            &cv,
        )
        .unwrap();
    for fold in fixed.folds() {
        assert_eq!(fold.path().lambda_start(), Some(3.0));
        close(fold.path().lambdas().unwrap()[1], 0.3);
    }
    let mae = model
        .cross_validate_path(
            &x,
            &y,
            &[1.0],
            &cv.metric(ValidationMetric::MeanAbsoluteError),
        )
        .unwrap();
    assert_eq!(mae.metric(), ValidationMetric::MeanAbsoluteError);
}

#[test]
fn binomial_irls_scores_fractional_responses_with_both_metrics() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
    let y = [0.1, 0.6, 0.3, 0.7, 0.9, 0.5];
    let folds = FoldAssignments::kfold(6, 3, 42).unwrap();
    let problem = Problem::new(&x, Binomial::new(&y), L1::new(1.0));
    let solver = shrinkage::Irls::new().tolerance(1e-9);
    for metric in [
        ValidationMetric::LogLoss,
        ValidationMetric::BinomialDeviance,
    ] {
        let cv = CrossValidation::new(folds.clone()).metric(metric);
        let result = problem
            .cross_validate_regularization_path_with(
                &solver,
                &PathOptions::new().nlambda(3).lambda_min_ratio(0.2),
                &cv,
            )
            .unwrap();
        for fold in result.folds() {
            let held = x.select_rows(fold.validation_rows()).unwrap();
            let held_y = fold
                .validation_rows()
                .iter()
                .map(|&i| y[i])
                .collect::<Vec<_>>();
            for (j, fit) in fold.path().fits().iter().enumerate() {
                close(
                    fold.scores()[j],
                    metric.score(&held_y, &fit.predict(&held).unwrap()).unwrap(),
                );
            }
        }
    }
}

struct RefitFailure(bool);

impl<M, P> shrinkage::PathSolver<Problem<&M, Gaussian<'_>, P>, L1> for RefitFailure
where
    M: shrinkage::lazymatrix::RawColumns<f64>
        + shrinkage::lazymatrix::ColumnStats<f64>
        + shrinkage::lazymatrix::MatVecInto<Vec<f64>>
        + shrinkage::lazymatrix::MatTransposeVecInto<Vec<f64>>,
{
    type Fit = shrinkage::ProximalFit;
    type Error = shrinkage::FitError<M::Error>;
    fn solve_path(
        &self,
        problem: &Problem<&M, Gaussian<'_>, P>,
        penalties: Vec<L1>,
    ) -> Result<shrinkage::PathFit<Self::Fit, L1>, Self::Error> {
        let mut solver = shrinkage::ProximalGradient::new().tolerance(1e-8);
        if problem.design().nrows() == 6 {
            if self.0 {
                return Err(shrinkage::FitError::NumericalFailure {
                    operation: "injecting a refit failure",
                    iteration: 0,
                });
            }
            solver = solver.max_iterations(1).initial_step(0.0001);
        }
        shrinkage::PathSolver::solve_path(&solver, problem, penalties)
    }
}

#[test]
fn refit_failures_and_iteration_limits_return_no_selected_model() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
    let y = [0.0, 1.0, 1.0, 3.0, 5.0, 4.0];
    let cv = CrossValidation::new(FoldAssignments::kfold(6, 3, 42).unwrap());
    let problem = Problem::new(&x, Gaussian::new(&y), L1::new(1.0));
    assert!(matches!(
        problem.cross_validate_path_with(&RefitFailure(true), vec![L1::new(0.1)], &cv),
        Err(CrossValidationError::Refit {
            source: shrinkage::FitError::NumericalFailure { .. }
        })
    ));
    assert!(matches!(
        problem.cross_validate_path_with(&RefitFailure(false), vec![L1::new(0.1)], &cv),
        Err(CrossValidationError::NonConverged {
            fold: None,
            candidate: 0
        })
    ));
}
