//! Automatic paths stop on deviance without truncating explicit candidates.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{Lasso, PathOptions, PathTermination};

#[test]
fn automatic_paths_stop_by_default_and_can_fit_the_full_sequence() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [-1.0, 1.0];
    let model = Lasso::new(1.0).tolerance(1e-10);
    let path = model
        .fit_regularization_path(&x, &y, &PathOptions::new())
        .unwrap();
    assert!(path.len() >= 5 && path.len() < 100);
    assert_eq!(path.termination(), PathTermination::DevianceRatio);
    assert_eq!(path.penalties().len(), path.len());
    assert_eq!(path.lambdas().unwrap().len(), path.len());
    assert_eq!(path.deviances().unwrap().len(), path.len());
    assert!((path.null_deviance().unwrap() - 1.0).abs() < 1e-14);
    let full = model
        .fit_regularization_path(&x, &y, &PathOptions::new().early_stopping(false))
        .unwrap();
    assert_eq!(full.len(), 100);
    assert_eq!(full.termination(), PathTermination::Completed);
    assert_eq!(
        path.lambdas().unwrap(),
        &full.lambdas().unwrap()[..path.len()]
    );
    let explicit = model.fit_path(&x, &y, full.lambdas().unwrap()).unwrap();
    assert_eq!(explicit.len(), 100);
    assert_eq!(explicit.termination(), PathTermination::Completed);
}

#[test]
fn every_solver_and_runtime_and_multivariate_paths_share_stopping() {
    use shrinkage::{
        AxisPenalty, CoefficientLayout, CoordinateDescent, ErasedPathPenalty, Gaussian, Hybrid,
        Irls, L1, MatrixDesign, MultivariateGaussian, PenaltyAxis, Problem, ProximalGradient,
        RuntimePathProblem, Slope,
    };
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [-1.0, 1.0];
    let options = PathOptions::new()
        .nlambda(12)
        .deviance_change_tolerance(1.0)
        .deviance_ratio_tolerance(1.0);
    let problem = Problem::new(&x, Gaussian::new(&y), L1::new(1.0));
    macro_rules! verify {
        ($path:expr) => {{
            let path = $path.unwrap();
            assert_eq!(path.len(), 5);
            assert_eq!(path.termination(), PathTermination::DevianceChange);
            assert_eq!(path.deviances().unwrap().len(), 5);
            path
        }};
    }
    let reference =
        verify!(problem.fit_regularization_path_with(&CoordinateDescent::new(), &options));
    let _ = verify!(problem.fit_regularization_path_with(&Irls::new(), &options));
    let proximal =
        verify!(problem.fit_regularization_path_with(&ProximalGradient::new(), &options));
    assert_eq!(proximal.lambdas(), reference.lambdas());
    let hybrid = Problem::new(&x, Gaussian::new(&y), Slope::new(vec![1.0]));
    let _ = verify!(hybrid.fit_regularization_path_with(&Hybrid::new(), &options));
    let runtime = RuntimePathProblem::new(
        Box::new(MatrixDesign::new(&x)),
        Box::new(Gaussian::new(&y)),
        Box::new(ErasedPathPenalty::new(L1::new(1.0))),
    );
    let _ = verify!(runtime.fit_regularization_path_with(&ProximalGradient::new(), &options));
    let y = [-1.0, 1.0, -2.0, 2.0];
    let layout = CoefficientLayout::new(1, 2).unwrap();
    let datafit = MultivariateGaussian::new(&y, 2, 2);
    let penalty = AxisPenalty::new(layout, PenaltyAxis::All, L1::new(1.0));
    let joint = Problem::new(&x, datafit, penalty.clone());
    let path = verify!(joint.fit_regularization_path_with(&ProximalGradient::new(), &options));
    let runtime = Problem::new(
        Box::new(MatrixDesign::new(&x)) as Box<dyn shrinkage::RuntimeDesign>,
        datafit,
        AxisPenalty::new(
            layout,
            PenaltyAxis::All,
            ErasedPathPenalty::new(L1::new(1.0)),
        ),
    );
    let other = verify!(runtime.fit_regularization_path_with(&ProximalGradient::new(), &options));
    assert_eq!(path.lambdas(), other.lambdas());
    for (a, b) in path
        .deviances()
        .unwrap()
        .iter()
        .zip(other.deviances().unwrap())
    {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn fractional_binomial_deviance_uses_the_saturated_loss() {
    use shrinkage::{Binomial, NullModelDatafit};
    let y = [0.2_f64, 0.5, 0.8, 0.0, 1.0];
    let eta = y.map(|y| {
        if y == 0.0 {
            -1000.0
        } else if y == 1.0 {
            1000.0
        } else {
            (y / (1.0 - y)).ln()
        }
    });
    assert!(Binomial::new(&y).deviance(&eta).unwrap().unwrap() < 1e-15);
    let y = [0.2, 0.8];
    let expected = 2.0 * (0.2_f64 * (0.2_f64 / 0.5).ln() + 0.8_f64 * (0.8_f64 / 0.5).ln());
    let actual = Binomial::new(&y).deviance(&[0.0; 2]).unwrap().unwrap();
    assert!((actual - expected).abs() < 1e-14);
    let binary = Binomial::new(&[0.0, 1.0]);
    assert!((binary.deviance(&[0.0; 2]).unwrap().unwrap() - 2.0 * 2.0_f64.ln()).abs() < 1e-14);
}

#[test]
fn invalid_stopping_options_are_rejected() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    for options in [
        PathOptions::new().min_points(0),
        PathOptions::new().deviance_change_tolerance(-1.0),
        PathOptions::new().deviance_change_tolerance(f64::NAN),
        PathOptions::new().deviance_ratio_tolerance(0.0),
        PathOptions::new().deviance_ratio_tolerance(1.1),
        PathOptions::new().deviance_ratio_tolerance(f64::INFINITY),
    ] {
        assert!(
            Lasso::new(1.0)
                .fit_regularization_path(&x, &[-1.0, 1.0], &options)
                .is_err()
        );
    }
}

#[test]
fn custom_losses_without_deviance_keep_the_full_sequence() {
    use shrinkage::{
        Datafit, FitError, Gaussian, L1, NullModelDatafit, Problem, ProximalGradient, SmoothDatafit,
    };
    struct Custom<'a>(Gaussian<'a>);
    impl Datafit for Custom<'_> {
        type Error = FitError;
        fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
            self.0.value(predictor)
        }
    }
    impl SmoothDatafit for Custom<'_> {
        fn nobs(&self) -> usize {
            self.0.nobs()
        }
        fn fits_intercept(&self) -> bool {
            self.0.fits_intercept()
        }
        fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
            self.0.gradient(predictor, output)
        }
    }
    impl NullModelDatafit for Custom<'_> {
        fn null_intercepts(&self) -> Result<Vec<f64>, FitError> {
            self.0.null_intercepts()
        }
    }
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let path = Problem::new(&x, Custom(Gaussian::new(&[-1.0, 1.0])), L1::new(1.0))
        .fit_regularization_path_with(&ProximalGradient::new(), &PathOptions::new().nlambda(8))
        .unwrap();
    assert_eq!(path.len(), 8);
    assert_eq!(path.termination(), PathTermination::Completed);
    assert!(path.null_deviance().is_none());
    assert!(path.deviances().is_none());
}

#[test]
fn null_models_and_short_sequences_do_not_divide_by_zero() {
    let x = Matrix::empty(2, 0);
    let model = Lasso::new(1.0);
    let options = PathOptions::new().nlambda(8);
    let path = model
        .fit_regularization_path(&x, &[2.0, 2.0], &options)
        .unwrap();
    assert_eq!(path.len(), 8);
    assert_eq!(path.null_deviance(), Some(0.0));
    assert_eq!(path.deviances().unwrap(), &[0.0; 8]);
    let path = model
        .fit_regularization_path(&x, &[1.0, 3.0], &options)
        .unwrap();
    assert_eq!(path.len(), 5);
    assert_eq!(path.termination(), PathTermination::DevianceChange);
    let path = model
        .fit_regularization_path(&x, &[1.0, 3.0], &options.nlambda(3))
        .unwrap();
    assert_eq!(path.len(), 3);
    assert_eq!(path.termination(), PathTermination::Completed);
}
