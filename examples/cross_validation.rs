//! Whole paths per fold, held-out selection, and original-scale full-data refits.

use faer::{Col, Mat};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, CrossValidation, FoldAssignments, Gaussian, Hybrid,
    Lasso, PathOptions, Problem, SelectionRule, Slope,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(12, 3, |i, j| match j {
        0 => i as f64,
        1 => (i % 3) as f64,
        _ => (i % 2) as f64,
    });
    let y = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 8.0, 7.0, 10.0, 9.0, 11.0];
    let binary = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let folds = FoldAssignments::stratified_binomial(&binary, 3, 42)?;
    let cv = CrossValidation::new(folds).selection_rule(SelectionRule::OneStandardError);
    let options = PathOptions::new().nlambda(12).lambda_min_ratio(0.05);
    let lasso = Lasso::new(1.0).cross_validate_regularization_path(&x, &y, &options, &cv)?;
    println!(
        "lasso candidate={}, full-data lambda={:.4}, validation MSE={:.4}",
        lasso.selected_index(),
        lasso.lambda().unwrap(),
        lasso.mean_scores()[lasso.selected_index()]
    );
    let predictions = lasso.fit().predict(&x)?;
    assert_eq!(predictions.len(), y.len());

    let solver = Hybrid::<Col<f64>>::new();
    macro_rules! show {
        ($family:ident, $response:expr, $penalty:expr) => {{
            let result = Problem::new(&x, $family::new($response), $penalty)
                .cross_validate_regularization_path_with(&solver, &options, &cv)?;
            let j = result.selected_index();
            println!(
                "{} candidate={j}, validation {:?}={:.4}, SE={:.4}",
                stringify!($family),
                result.metric(),
                result.mean_scores()[j],
                result.standard_errors()[j]
            );
            assert_eq!(result.fit().predict(&x)?.len(), y.len());
            for fold in result.folds() {
                assert_eq!(
                    fold.path().len(),
                    result.relative_strengths().unwrap().len()
                );
            }
        }};
    }
    show!(Gaussian, &y, ClusteredLasso::new(1.0, 0.2));
    show!(Gaussian, &y, ChainFusedLasso::new(1.0, 0.2));
    show!(Gaussian, &y, Slope::new(vec![1.0, 0.6, 0.2]));
    show!(Binomial, &binary, ClusteredLasso::new(1.0, 0.2));
    show!(Binomial, &binary, ChainFusedLasso::new(1.0, 0.2));
    show!(Binomial, &binary, Slope::new(vec![1.0, 0.6, 0.2]));

    // The caller orders independent strengths by preference for simpler fits.
    let explicit = Problem::new(&x, Gaussian::new(&y), ClusteredLasso::new(1.0, 1.0))
        .cross_validate_path_with(
            &solver,
            vec![
                ClusteredLasso::new(0.5, 0.2),
                ClusteredLasso::new(0.1, 0.3),
                ClusteredLasso::new(0.1, 0.02),
            ],
            &cv,
        )?;
    println!(
        "selected L1={:.4}, fusion={:.4}",
        explicit.penalty().l1(),
        explicit.penalty().fusion()
    );
    Ok(())
}
