//! Runtime capability preservation, preparation reuse, and backend failures.

#[path = "common/matrix.rs"]
mod matrix;

use std::{cell::Cell, rc::Rc};

use matrix::Matrix;
use shrinkage::{
    AxisPenalty, BackendError, Binomial, CoefficientLayout, ErasedPathPenalty, FitError, Gaussian,
    L1, MatrixDesign, MultivariateGaussian, Normalization, PathOptions, PenaltyAxis, Preprocessing,
    Problem, ProximalDesign, ProximalGradient, ProximalPenalty, RuntimeDesign, RuntimePathProblem,
    RuntimeProblem, Slope, Termination,
};

struct Counted<'a> {
    design: MatrixDesign<'a, Matrix>,
    preparations: Rc<Cell<usize>>,
    products: Rc<Cell<usize>>,
    fail_after: Option<usize>,
}

impl RuntimeDesign for Counted<'_> {
    fn nrows(&self) -> usize {
        self.design.nrows()
    }
    fn ncols(&self) -> usize {
        self.design.ncols()
    }
    fn prepare(
        &self,
        normalization: Normalization,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        intercept: bool,
    ) -> Result<Box<dyn ProximalDesign<Error = BackendError> + '_>, FitError<BackendError>> {
        self.preparations.set(self.preparations.get() + 1);
        Ok(Box::new(CountedProducts {
            design: self
                .design
                .prepare(normalization, centers, scales, intercept)?,
            products: self.products.clone(),
            fail_after: self.fail_after,
        }))
    }
}

struct CountedProducts<'a> {
    design: Box<dyn ProximalDesign<Error = BackendError> + 'a>,
    products: Rc<Cell<usize>>,
    fail_after: Option<usize>,
}

impl CountedProducts<'_> {
    fn check(&self, output: &mut [f64]) -> Result<(), FitError<BackendError>> {
        self.products.set(self.products.get() + 1);
        if self
            .fail_after
            .is_some_and(|limit| self.products.get() > limit)
        {
            output.fill(f64::NAN);
            return Err(FitError::Backend {
                operation: "injected path product",
                source: BackendError::new(std::io::Error::other("injected failure")),
            });
        }
        Ok(())
    }
}

impl ProximalDesign for CountedProducts<'_> {
    type Error = BackendError;
    fn nrows(&self) -> usize {
        self.design.nrows()
    }
    fn ncols(&self) -> usize {
        self.design.ncols()
    }
    fn forward(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), FitError<BackendError>> {
        self.check(output)?;
        self.design.forward(input, output)
    }
    fn transpose(
        &mut self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FitError<BackendError>> {
        self.check(output)?;
        self.design.transpose(input, output)
    }
    fn into_preprocessing(self: Box<Self>) -> Preprocessing {
        self.design.into_preprocessing()
    }
}

#[test]
fn runtime_automatic_paths_prepare_once_and_match_typed_paths() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-1.0, -7.0, 9.0, 3.0];
    let preparations = Rc::new(Cell::new(0));
    let runtime = RuntimePathProblem::new(
        Box::new(Counted {
            design: MatrixDesign::new(&x),
            preparations: preparations.clone(),
            products: Rc::new(Cell::new(0)),
            fail_after: None,
        }),
        Box::new(Gaussian::new(&y)),
        Box::new(ErasedPathPenalty::new(Slope::new(vec![2.0, 1.0]))),
    );
    let typed = Problem::new(&x, Gaussian::new(&y), Slope::new(vec![2.0, 1.0]));
    let options = PathOptions::new().nlambda(4).lambda_min_ratio(0.01);
    let solver = ProximalGradient::new().tolerance(1e-10);
    let a = runtime
        .fit_regularization_path_with(&solver, &options)
        .unwrap();
    let b = typed
        .fit_regularization_path_with(&solver, &options)
        .unwrap();
    assert_eq!(preparations.get(), 1);
    assert_eq!(a.lambdas(), b.lambdas());
    assert_eq!(a.maximum_penalty(), b.maximum_penalty());
    for (a, b) in a.fits().iter().zip(b.fits()) {
        assert_eq!(a.coefficients(), b.coefficients());
        assert_eq!(a.intercept(), b.intercept());
        assert_eq!(a.objective(), b.objective());
        assert_eq!(a.iterations(), b.iterations());
    }
    assert_eq!(
        runtime.fit_with(&solver).unwrap().termination(),
        Termination::Converged
    );
}

#[test]
fn runtime_binomial_path_has_the_same_null_start_as_typed() {
    let x = Matrix::from_rows(&[&[-2.0], &[0.0], &[1.0], &[3.0]]);
    let y = [0.0, 0.0, 1.0, 0.0];
    let runtime = RuntimePathProblem::new(
        Box::new(MatrixDesign::new(&x)),
        Box::new(Binomial::new(&y)),
        Box::new(ErasedPathPenalty::new(L1::new(1.0))),
    )
    .normalize(Normalization::None);
    let typed = Problem::new(&x, Binomial::new(&y), L1::new(1.0)).normalize(Normalization::None);
    let options = PathOptions::new().nlambda(3).lambda_min_ratio(0.1);
    let solver = ProximalGradient::new().tolerance(1e-9);
    let a = runtime
        .fit_regularization_path_with(&solver, &options)
        .unwrap();
    let b = typed
        .fit_regularization_path_with(&solver, &options)
        .unwrap();
    for (a, b) in a.fits().iter().zip(b.fits()) {
        assert_eq!(a.coefficients(), b.coefficients());
        assert_eq!(a.intercept(), b.intercept());
        assert_eq!(a.termination(), Termination::Converged);
    }
}

#[test]
fn runtime_multivariate_paths_keep_all_penalty_axes() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-5.0, 1.0, 1.0, 7.0, -6.0, -6.0, 6.0, 6.0];
    let layout = CoefficientLayout::new(2, 2).unwrap();
    let options = PathOptions::new().nlambda(3).lambda_min_ratio(0.1);
    let solver = ProximalGradient::new().tolerance(1e-9);
    for axis in [
        PenaltyAxis::All,
        PenaltyAxis::Features,
        PenaltyAxis::Responses,
    ] {
        let weights = if axis == PenaltyAxis::All {
            vec![2.0, 1.0, 1.0, 0.0]
        } else {
            vec![2.0, 1.0]
        };
        let runtime = Problem::new(
            Box::new(MatrixDesign::new(&x)) as Box<dyn RuntimeDesign>,
            MultivariateGaussian::new(&y, 4, 2),
            AxisPenalty::new(
                layout,
                axis,
                ErasedPathPenalty::new(Slope::new(weights.clone())),
            ),
        );
        let typed = Problem::new(
            &x,
            MultivariateGaussian::new(&y, 4, 2),
            AxisPenalty::new(layout, axis, Slope::new(weights)),
        );
        let a = runtime
            .fit_regularization_path_with(&solver, &options)
            .unwrap();
        let b = typed
            .fit_regularization_path_with(&solver, &options)
            .unwrap();
        assert_eq!(a.maximum_penalty(), b.maximum_penalty());
        for (a, b) in a.fits().iter().zip(b.fits()) {
            assert_eq!(a.coefficients(), b.coefficients());
            assert_eq!(a.intercepts(), b.intercepts());
            assert_eq!(a.termination(), Termination::Converged);
            assert!(a.stopping_value() <= a.stopping_threshold());
        }
        let point = AxisPenalty::new(
            layout,
            axis,
            Box::new(L1::new(0.5)) as Box<dyn ProximalPenalty>,
        );
        let explicit = runtime.fit_path_with(&solver, vec![point]).unwrap();
        assert_eq!(explicit.len(), 1);
    }
}

#[test]
fn backend_failures_at_null_preparation_and_later_points_preserve_sources() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [-3.0, 3.0];
    for fail_after in [0, 8] {
        let products = Rc::new(Cell::new(0));
        let problem = RuntimePathProblem::new(
            Box::new(Counted {
                design: MatrixDesign::new(&x),
                preparations: Rc::new(Cell::new(0)),
                products: products.clone(),
                fail_after: Some(fail_after),
            }),
            Box::new(Gaussian::new(&y)),
            Box::new(ErasedPathPenalty::new(L1::new(1.0))),
        );
        let result = problem.fit_regularization_path_with(
            &ProximalGradient::new(),
            &PathOptions::new().nlambda(3).lambda_min_ratio(0.01),
        );
        match result {
            Err(FitError::Backend { operation, source }) => {
                assert_eq!(operation, "injected path product");
                assert!(source.as_error().is::<std::io::Error>());
                assert_eq!(source.to_string(), "injected failure");
            }
            _ => panic!("expected the original backend error"),
        }
        assert_eq!(products.get(), fail_after + 1);
    }
}

#[test]
fn explicit_runtime_paths_accept_mixed_penalties_without_cloning() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let problem = RuntimeProblem::new(
        Box::new(MatrixDesign::new(&x)),
        Box::new(Gaussian::new(&[-2.0, 2.0])),
        Box::new(L1::new(99.0)),
    );
    let penalties: Vec<Box<dyn ProximalPenalty>> =
        vec![Box::new(L1::new(1.0)), Box::new(shrinkage::Ridge::new(1.0))];
    let path = problem
        .fit_path_with(&ProximalGradient::new(), penalties)
        .unwrap();
    assert_eq!(path.fits()[0].coefficients(), &[1.0]);
    assert_eq!(path.fits()[1].coefficients(), &[1.0]);
    let empty = Matrix::empty(2, 0);
    let problem = RuntimePathProblem::new(
        Box::new(MatrixDesign::new(&empty)),
        Box::new(Gaussian::new(&[1.0, 3.0])),
        Box::new(ErasedPathPenalty::new(L1::new(1.0))),
    );
    assert!(
        problem
            .fit_regularization_path_with(&ProximalGradient::new(), &PathOptions::new().nlambda(1))
            .unwrap()
            .fits()[0]
            .coefficients()
            .is_empty()
    );
}

#[test]
fn incompatible_runtime_path_solver_rejects_before_preparation() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let preparations = Rc::new(Cell::new(0));
    let problem = RuntimePathProblem::new(
        Box::new(Counted {
            design: MatrixDesign::new(&x),
            preparations: preparations.clone(),
            products: Rc::new(Cell::new(0)),
            fail_after: None,
        }),
        Box::new(Gaussian::new(&[-2.0, 2.0])),
        Box::new(ErasedPathPenalty::new(L1::new(1.0))),
    );
    let solver = shrinkage::CoordinateDescent::new();
    assert!(matches!(
        problem.fit_path_with(&solver, vec![L1::new(1.0)]),
        Err(FitError::UnsupportedCombination { .. })
    ));
    assert!(matches!(
        problem.fit_regularization_path_with(&solver, &PathOptions::new()),
        Err(FitError::UnsupportedCombination { .. })
    ));
    assert_eq!(preparations.get(), 0);
    let solver = shrinkage::Hybrid::<Vec<f64>>::new();
    assert!(matches!(
        problem.fit_path_with(&solver, vec![L1::new(1.0)]),
        Err(FitError::UnsupportedCombination { .. })
    ));
    assert!(matches!(
        problem.fit_regularization_path_with(&solver, &PathOptions::new()),
        Err(FitError::UnsupportedCombination { .. })
    ));
    assert_eq!(preparations.get(), 0);
}
