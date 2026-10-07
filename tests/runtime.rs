//! Independent runtime selection of proximal components.

#[path = "common/matrix.rs"]
mod matrix;

use shrinkage::{
    CoordinateDescent, Datafit, ElasticNet, FitError, Gaussian, L1, MatrixDesign, Normalization,
    Penalty, Problem, ProximalFit, ProximalGradient, ProximalPenalty, Ridge, RuntimeDesign,
    RuntimeProblem, SmoothDatafit, StoppingCriterion, Termination,
};

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
struct CoupledQuadratic;

impl CoupledQuadratic {
    fn validate(&self, nfeatures: usize) -> Result<(), FitError> {
        if nfeatures != 2 {
            return Err(FitError::UnsupportedCombination {
                solver: "ProximalGradient",
                reason: format!("CoupledQuadratic requires two features, received {nfeatures}"),
                suggestion: "supply two features or select Ridge for an arbitrary feature count",
            });
        }
        Ok(())
    }
}

impl Penalty for CoupledQuadratic {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        self.validate(coefficients.len())?;
        Ok(0.5 * (coefficients[0] - coefficients[1]).powi(2))
    }
}

impl ProximalPenalty for CoupledQuadratic {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.validate(input.len())?;
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

fn assert_same_fit(runtime: &ProximalFit, typed: &ProximalFit, prediction_design: &matrix::Matrix) {
    assert_eq!(runtime.coefficients(), typed.coefficients());
    assert_eq!(runtime.intercept(), typed.intercept());
    assert_eq!(runtime.objective(), typed.objective());
    assert_eq!(runtime.termination(), typed.termination());
    assert_eq!(runtime.iterations(), typed.iterations());
    assert_eq!(runtime.stopping_criterion(), typed.stopping_criterion());
    assert_eq!(runtime.stopping_value(), typed.stopping_value());
    assert_eq!(runtime.stopping_threshold(), typed.stopping_threshold());
    assert_eq!(runtime.step_size(), typed.step_size());
    assert_eq!(
        runtime.preprocessing().centering(),
        typed.preprocessing().centering()
    );
    assert_eq!(
        runtime.preprocessing().scaling(),
        typed.preprocessing().scaling()
    );
    assert_eq!(
        runtime.preprocessing().centers(),
        typed.preprocessing().centers()
    );
    assert_eq!(
        runtime.preprocessing().scales(),
        typed.preprocessing().scales()
    );
    assert_eq!(
        runtime.preprocessing().centers_were_supplied(),
        typed.preprocessing().centers_were_supplied()
    );
    assert_eq!(
        runtime.preprocessing().scales_were_supplied(),
        typed.preprocessing().scales_were_supplied()
    );
    assert_eq!(
        runtime.predict(prediction_design).unwrap(),
        typed.predict(prediction_design).unwrap()
    );
}

fn compare_components<D, P>(
    x: &matrix::Matrix,
    datafit: D,
    penalty: P,
    normalization: Normalization,
    supplied: bool,
    max_iterations: usize,
) where
    D: SmoothDatafit + Clone,
    P: ProximalPenalty + Clone,
{
    let solver = ProximalGradient::<Vec<f64>>::new()
        .tolerance(1e-9)
        .max_iterations(max_iterations);
    let mut typed = Problem::new(x, datafit.clone(), penalty.clone()).normalize(normalization);
    let mut runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(x)),
        Box::new(datafit),
        Box::new(penalty),
    )
    .normalize(normalization);
    if supplied {
        typed = typed
            .with_centers(vec![3.0, 1.25])
            .with_scales(vec![2.0, 0.5]);
        runtime = runtime
            .with_centers(vec![3.0, 1.25])
            .with_scales(vec![2.0, 0.5]);
    }
    let typed = typed.fit_with(&solver).unwrap();
    assert_eq!(
        typed.termination(),
        if max_iterations == 1 {
            Termination::IterationLimit
        } else {
            Termination::Converged
        }
    );
    let held_out = matrix::Matrix::from_rows(&[&[2.0, -1.0], &[0.0, 0.0], &[-3.0, 4.0]]);
    for _ in 0..2 {
        let fit = runtime.fit_with(&solver).unwrap();
        assert_same_fit(&fit, &typed, &held_out);
        assert_eq!(fit.predict(x).unwrap(), typed.predict(x).unwrap());
    }
}

#[test]
fn typed_and_runtime_fits_agree_for_builtin_and_external_components() {
    let x = matrix::Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    for intercept in [false, true] {
        for normalization in [
            Normalization::Auto,
            Normalization::None,
            Normalization::Center,
            Normalization::Standardize,
        ] {
            for supplied in [false, true] {
                for max_iterations in [1, 10_000] {
                    let gaussian = Gaussian::new(&y).fit_intercept(intercept);
                    macro_rules! compare {
                        ($datafit:expr) => {
                            compare_components(
                                &x,
                                $datafit,
                                L1::new(0.2),
                                normalization,
                                supplied,
                                max_iterations,
                            );
                            compare_components(
                                &x,
                                $datafit,
                                Ridge::new(0.5),
                                normalization,
                                supplied,
                                max_iterations,
                            );
                            compare_components(
                                &x,
                                $datafit,
                                ElasticNet::new(0.2, 0.5),
                                normalization,
                                supplied,
                                max_iterations,
                            );
                            compare_components(
                                &x,
                                $datafit,
                                CoupledQuadratic,
                                normalization,
                                supplied,
                                max_iterations,
                            );
                        };
                    }
                    compare!(gaussian);
                    compare!(TwiceGaussian(gaussian));
                }
            }
        }
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
        let typed = Problem::new(
            &empty,
            Gaussian::new(&[2.0, 3.0, 4.0]).fit_intercept(intercept),
            Ridge::new(1.0),
        )
        .fit_with(&ProximalGradient::<Vec<f64>>::new())
        .unwrap();
        let fit = RuntimeProblem::new(
            Box::new(MatrixDesign::<_, Vec<f64>>::new(&empty)),
            Box::new(Gaussian::new(&[2.0, 3.0, 4.0]).fit_intercept(intercept)),
            Box::new(Ridge::new(1.0)),
        )
        .fit_with(&ProximalGradient::new())
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!(fit.coefficients().is_empty());
        assert_same_fit(&fit, &typed, &empty);
        assert!((fit.intercept() - if intercept { 3.0 } else { 0.0 }).abs() < 1e-6);
    }
}

#[test]
fn unsupported_stopping_rules_report_compatibility_and_a_supported_alternative() {
    let x = matrix::Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0, 3.0, 5.0];
    for criterion in [
        StoppingCriterion::kkt_violation(1e-6),
        StoppingCriterion::duality_gap(1e-6),
    ] {
        let solver = ProximalGradient::<Vec<f64>>::new().terminate_on(criterion);
        let typed = Problem::new(&x, Gaussian::new(&y), Ridge::new(0.1))
            .fit_with(&solver)
            .unwrap_err();
        let runtime = RuntimeProblem::new(
            Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
            Box::new(Gaussian::new(&y)),
            Box::new(Ridge::new(0.1)),
        )
        .fit_with(&solver)
        .unwrap_err();
        assert_eq!(typed.to_string(), runtime.to_string());
        assert!(matches!(
            runtime,
            FitError::UnsupportedCombination {
                solver: "ProximalGradient",
                ..
            }
        ));
        assert!(runtime.to_string().contains("unsupported fit combination"));
        assert!(runtime.to_string().contains("proximal_gradient_mapping"));
        assert_eq!(x.visits.get(), 0);
    }
    let error = Problem::new(&x, Gaussian::new(&y), L1::new(0.1))
        .fit_with(
            &CoordinateDescent::new()
                .terminate_on(StoppingCriterion::proximal_gradient_mapping(1e-6)),
        )
        .unwrap_err();
    assert!(error.to_string().contains("unsupported fit combination"));
    assert!(matches!(
        error,
        FitError::UnsupportedCombination {
            solver: "CoordinateDescent",
            ..
        }
    ));
    assert!(error.to_string().contains("KKT violation or duality gap"));
    assert_eq!(x.visits.get(), 0);
}

#[test]
fn runtime_coordinate_selection_reports_missing_capabilities_before_preparation() {
    let x = matrix::Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-4.0, -2.0, 2.0, 4.0];
    for doubled in [false, true] {
        for penalty_id in 0..4 {
            let problem = RuntimeProblem::new(
                Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
                select_datafit(&y, doubled),
                select_penalty(penalty_id),
            );
            x.visits.set(0);
            let error = problem.fit_with(&CoordinateDescent::new()).unwrap_err();
            assert!(matches!(
                error,
                FitError::UnsupportedCombination {
                    solver: "CoordinateDescent",
                    ..
                }
            ));
            let message = error.to_string();
            assert!(message.contains("coordinate updates"));
            assert!(message.contains("ProximalGradient<Vec<f64>>"));
            assert!(message.contains("typed Problem with Gaussian and L1"));
            assert_eq!(x.visits.get(), 0);
            assert_eq!(
                problem
                    .fit_with(&ProximalGradient::new())
                    .unwrap()
                    .termination(),
                Termination::Converged
            );
        }
    }
    assert_eq!(
        Problem::new(&x, Gaussian::new(&y), L1::new(0.2))
            .fit_with(&CoordinateDescent::new())
            .unwrap()
            .termination(),
        Termination::Converged
    );
}

#[test]
fn external_component_compatibility_errors_survive_runtime_dispatch() {
    let x = matrix::Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0, 3.0, 5.0];
    let solver = ProximalGradient::<Vec<f64>>::new();
    let typed = Problem::new(&x, Gaussian::new(&y), CoupledQuadratic)
        .fit_with(&solver)
        .unwrap_err();
    let runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
        Box::new(Gaussian::new(&y)),
        Box::new(CoupledQuadratic),
    )
    .fit_with(&solver)
    .unwrap_err();
    assert_eq!(typed.to_string(), runtime.to_string());
    assert!(
        matches!(runtime, FitError::UnsupportedCombination { solver: "ProximalGradient", ref reason, suggestion } if reason.contains("requires two features, received 1") && suggestion.contains("select Ridge"))
    );
    assert_eq!(
        Problem::new(&x, Gaussian::new(&y), Ridge::new(0.5))
            .fit_with(&solver)
            .unwrap()
            .termination(),
        Termination::Converged
    );
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
