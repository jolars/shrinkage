//! Independent runtime selection of proximal components.

#[path = "common/matrix.rs"]
mod matrix;

use shrinkage::{
    Datafit, ElasticNet, FitError, Gaussian, L1, MatrixDesign, Normalization, Penalty, Problem,
    ProximalGradient, ProximalPenalty, Ridge, RuntimeDesign, RuntimeProblem, SmoothDatafit,
    Termination,
};

struct TwiceGaussian<'a>(Gaussian<'a>);

impl Datafit for TwiceGaussian<'_> {
    type Error = FitError;

    fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
        Ok(2.0 * self.0.value(predictor)?)
    }
}

impl SmoothDatafit for TwiceGaussian<'_> {
    fn nobs(&self) -> usize {
        self.0.nobs()
    }

    fn fits_intercept(&self) -> bool {
        self.0.fits_intercept()
    }

    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.0.gradient(predictor, output)?;
        for value in output {
            *value *= 2.0;
        }
        Ok(())
    }
}

struct CoupledQuadratic;

impl Penalty for CoupledQuadratic {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        Ok(0.5 * (coefficients[0] - coefficients[1]).powi(2))
    }
}

impl ProximalPenalty for CoupledQuadratic {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        let mean = 0.5 * (input[0] + input[1]);
        let difference = 0.5 * (input[0] - input[1]) / (1.0 + 2.0 * step);
        output[0] = mean + difference;
        output[1] = mean - difference;
        Ok(())
    }
}

fn select_datafit(y: &[f64], doubled: bool) -> Box<dyn SmoothDatafit + '_> {
    let gaussian = Gaussian::new(y).fit_intercept(false);
    if doubled {
        Box::new(TwiceGaussian(gaussian))
    } else {
        Box::new(gaussian)
    }
}

fn select_penalty(penalty_id: usize) -> Box<dyn ProximalPenalty> {
    match penalty_id {
        0 => Box::new(L1::new(0.2)),
        1 => Box::new(Ridge::new(0.5)),
        2 => Box::new(ElasticNet::new(0.2, 0.5)),
        _ => Box::new(CoupledQuadratic),
    }
}

fn expected_coefficients(doubled: bool, penalty_id: usize) -> [f64; 2] {
    let weight = if doubled { 2.0 } else { 1.0 };
    match penalty_id {
        0 => [3.0 - 0.2 / weight, 1.0 - 0.2 / weight],
        1 => [3.0 * weight / (weight + 0.5), weight / (weight + 0.5)],
        2 => [
            (3.0 * weight - 0.2) / (weight + 0.5),
            (weight - 0.2) / (weight + 0.5),
        ],
        _ => [2.0 + weight / (weight + 2.0), 2.0 - weight / (weight + 2.0)],
    }
}

#[test]
fn components_are_selected_independently_before_problem_construction() {
    let x = matrix::Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-4.0, -2.0, 2.0, 4.0];
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-9);
    for doubled in [false, true] {
        for penalty_id in 0..4 {
            let design: Box<dyn RuntimeDesign> = Box::new(MatrixDesign::<_, Vec<f64>>::new(&x));
            let datafit = select_datafit(&y, doubled);
            let penalty = select_penalty(penalty_id);
            let fit = RuntimeProblem::new(design, datafit, penalty)
                .normalize(Normalization::None)
                .fit_with(&solver)
                .unwrap();
            assert_eq!(fit.termination(), Termination::Converged);
            let expected = expected_coefficients(doubled, penalty_id);
            for (&actual, expected) in fit.coefficients().iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
            }
        }
    }
}

#[test]
fn runtime_normalization_and_intercept_preserve_typed_parameters() {
    let x = matrix::Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-9);
    for intercept in [false, true] {
        let datafit = Gaussian::new(&y).fit_intercept(intercept);
        let penalty = ElasticNet::new(0.15, 0.3);
        let typed = Problem::new(&x, datafit, penalty)
            .with_centers(vec![3.0, 1.25])
            .with_scales(vec![2.0, 0.5])
            .fit_with(&solver)
            .unwrap();
        let problem = RuntimeProblem::new(
            Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
            Box::new(datafit),
            Box::new(penalty),
        )
        .with_centers(vec![3.0, 1.25])
        .with_scales(vec![2.0, 0.5]);
        for runtime in [
            problem.fit_with(&solver).unwrap(),
            problem.fit_with(&solver).unwrap(),
        ] {
            assert_eq!(runtime.coefficients(), typed.coefficients());
            assert_eq!(runtime.intercept(), typed.intercept());
            assert_eq!(runtime.objective(), typed.objective());
            assert_eq!(runtime.iterations(), typed.iterations());
            assert_eq!(runtime.stopping_value(), typed.stopping_value());
            assert_eq!(runtime.predict(&x).unwrap(), typed.predict(&x).unwrap());
            assert_eq!(
                runtime.preprocessing().centers(),
                typed.preprocessing().centers()
            );
            assert_eq!(
                runtime.preprocessing().scales(),
                typed.preprocessing().scales()
            );
            assert!(runtime.preprocessing().centers_were_supplied());
            assert!(runtime.preprocessing().scales_were_supplied());
        }
    }
}

#[test]
fn runtime_preparation_rejects_invalid_dimensions_entries_and_supplied_vectors() {
    let x = matrix::Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]);
    for y in [&[1.0, 3.0][..], &[1.0, f64::NAN, 5.0][..]] {
        let error = RuntimeProblem::new(
            Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
            Box::new(Gaussian::new(y)),
            Box::new(Ridge::new(0.1)),
        )
        .fit_with(&ProximalGradient::new())
        .unwrap_err();
        assert!(matches!(error, FitError::InvalidInput { .. }));
    }
    for scales in [vec![], vec![0.0], vec![f64::INFINITY]] {
        assert!(matches!(
            RuntimeProblem::new(
                Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
                Box::new(Gaussian::new(&[1.0, 3.0, 5.0])),
                Box::new(Ridge::new(0.1)),
            )
            .with_scales(scales)
            .fit_with(&ProximalGradient::new()),
            Err(FitError::InvalidInput { .. })
        ));
    }
    let invalid = matrix::Matrix::from_rows(&[&[f64::NAN]]);
    assert!(matches!(
        RuntimeProblem::new(
            Box::new(MatrixDesign::<_, Vec<f64>>::new(&invalid)),
            Box::new(Gaussian::new(&[1.0])),
            Box::new(Ridge::new(0.1)),
        )
        .fit_with(&ProximalGradient::new()),
        Err(FitError::InvalidInput { .. })
    ));
}

#[test]
fn runtime_design_supports_empty_feature_vectors() {
    let empty = matrix::Matrix::empty(3, 0);
    for intercept in [false, true] {
        let fit = RuntimeProblem::new(
            Box::new(MatrixDesign::<_, Vec<f64>>::new(&empty)),
            Box::new(Gaussian::new(&[2.0, 3.0, 4.0]).fit_intercept(intercept)),
            Box::new(Ridge::new(1.0)),
        )
        .fit_with(&ProximalGradient::new())
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!(fit.coefficients().is_empty());
        assert!((fit.intercept() - if intercept { 3.0 } else { 0.0 }).abs() < 1e-6);
    }
}

#[cfg(all(feature = "faer_v0_24", feature = "ndarray_v0_17"))]
#[test]
fn runtime_backend_selection_does_not_select_a_datafit_or_penalty() {
    let rows = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]];
    let dense = faer::Mat::from_fn(4, 2, |i, j| rows[i][j]);
    let array = ndarray::Array2::from_shape_fn((4, 2), |(i, j)| rows[i][j]);
    let y = [-4.0, -2.0, 2.0, 4.0];
    for backend in 0..2 {
        for doubled in [false, true] {
            for penalty_id in 0..4 {
                let design: Box<dyn RuntimeDesign> = match backend {
                    0 => Box::new(MatrixDesign::<_, faer::Col<f64>>::new(&dense)),
                    _ => Box::new(MatrixDesign::<_, ndarray::Array1<f64>>::new(&array)),
                };
                let datafit = select_datafit(&y, doubled);
                let penalty = select_penalty(penalty_id);
                let fit = RuntimeProblem::new(design, datafit, penalty)
                    .normalize(Normalization::None)
                    .fit_with(&ProximalGradient::new().tolerance(1e-9))
                    .unwrap();
                assert_eq!(fit.termination(), Termination::Converged);
                for (&actual, expected) in fit
                    .coefficients()
                    .iter()
                    .zip(expected_coefficients(doubled, penalty_id))
                {
                    assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
                }
            }
        }
    }
}
