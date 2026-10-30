//! Explicit coefficient axes and joint Gaussian fitting.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    AxisPenalty, CoefficientLayout, Datafit, ElasticNet, FitError, Gaussian, L1, MatrixDesign,
    MultivariateGaussian, Normalization, Penalty, PenaltyAxis, Problem, ProximalGradient,
    ProximalPenalty, Ridge, RuntimeMultivariateProblem, StoppingCriterion, Termination,
};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
}

#[test]
fn loss_sums_responses_and_gradients_use_the_observation_count() {
    let y = [1.0, -2.0, 3.0, 4.0, -1.0, 2.0];
    let predictor = [0.0, 1.0, -1.0, 2.0, 3.0, 0.0];
    let datafit = MultivariateGaussian::new(&y, 2, 3);
    let expected: f64 = y
        .chunks_exact(2)
        .zip(predictor.chunks_exact(2))
        .map(|(y, eta)| Gaussian::new(y).value(eta).unwrap())
        .sum();
    close(datafit.value(&predictor).unwrap(), expected);
    let mut gradient = [0.0; 6];
    datafit.gradient(&predictor, &mut gradient).unwrap();
    for i in 0..predictor.len() {
        close(gradient[i], (predictor[i] - y[i]) / 2.0);
        let mut plus = predictor;
        let mut minus = predictor;
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        close(
            gradient[i],
            (datafit.value(&plus).unwrap() - datafit.value(&minus).unwrap()) / 2e-5,
        );
    }
}

// An external coupled penalty makes axis mistakes observable. Squared L2 and
// elementwise L1 would give the same result under every grouping.
#[derive(Clone, Copy)]
struct EuclideanNorm(f64);

impl Penalty for EuclideanNorm {
    type Error = FitError;
    fn value(&self, input: &[f64]) -> Result<f64, FitError> {
        Ok(self.0 * input.iter().map(|x| x * x).sum::<f64>().sqrt())
    }
}

impl ProximalPenalty for EuclideanNorm {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        let norm = input.iter().map(|x| x * x).sum::<f64>().sqrt();
        let shrink = if norm == 0.0 {
            0.0
        } else {
            (1.0 - step * self.0 / norm).max(0.0)
        };
        for (&entry, target) in input.iter().zip(output) {
            *target = entry * shrink;
        }
        Ok(())
    }
}

#[test]
fn penalty_axes_change_groups_and_the_exact_proximal_map() {
    let layout = CoefficientLayout::new(2, 2).unwrap();
    let input = [3.0, 4.0, 2.0, -1.0];
    let cases = [
        (
            PenaltyAxis::All,
            30.0_f64.sqrt(),
            [
                3.0 * (1.0 - 1.0 / 30.0_f64.sqrt()),
                4.0 * (1.0 - 1.0 / 30.0_f64.sqrt()),
                2.0 * (1.0 - 1.0 / 30.0_f64.sqrt()),
                -(1.0 - 1.0 / 30.0_f64.sqrt()),
            ],
        ),
        (
            PenaltyAxis::Features,
            5.0 + 5.0_f64.sqrt(),
            [
                2.4,
                3.2,
                2.0 * (1.0 - 1.0 / 5.0_f64.sqrt()),
                -(1.0 - 1.0 / 5.0_f64.sqrt()),
            ],
        ),
        (
            PenaltyAxis::Responses,
            13.0_f64.sqrt() + 17.0_f64.sqrt(),
            [
                3.0 * (1.0 - 1.0 / 13.0_f64.sqrt()),
                4.0 * (1.0 - 1.0 / 17.0_f64.sqrt()),
                2.0 * (1.0 - 1.0 / 13.0_f64.sqrt()),
                -(1.0 - 1.0 / 17.0_f64.sqrt()),
            ],
        ),
    ];
    for (axis, value, expected) in cases {
        let term = AxisPenalty::new(layout, axis, EuclideanNorm(1.0));
        assert_eq!(term.axis(), axis);
        close(term.penalty().0, 1.0);
        close(term.value(&input).unwrap(), value);
        let mut output = [0.0; 4];
        term.prox(&input, 1.0, &mut output).unwrap();
        for (&actual, expected) in output.iter().zip(expected) {
            close(actual, expected);
        }
        // The proximal first-order condition is exact away from zero groups.
        let displacement: Vec<_> = input.iter().zip(output).map(|(a, b)| a - b).collect();
        close(
            term.value(&displacement).unwrap(),
            match axis {
                PenaltyAxis::All => 1.0,
                _ => 2.0,
            },
        );
    }
}

#[test]
fn response_axis_penalty_fits_coupled_coefficients_jointly() {
    let x = Matrix::from_rows(&[&[-1.0, 0.0], &[1.0, 0.0], &[0.0, -1.0], &[0.0, 1.0]]);
    let y = [-3.0, 3.0, 0.0, 0.0, -4.0, 4.0, -0.5, 0.5];
    let layout = CoefficientLayout::new(2, 2).unwrap();
    let datafit = MultivariateGaussian::new(&y, 4, 2).fit_intercept(false);
    let term = AxisPenalty::new(layout, PenaltyAxis::Responses, EuclideanNorm(0.5));
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    let fit = Problem::new(&x, datafit, term.clone())
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    for (&actual, expected) in fit.coefficients().iter().zip([2.4, 0.0, 3.2, 0.0]) {
        close(actual, expected);
    }
    close(
        fit.objective(),
        datafit.value(&fit.predict(&x).unwrap()).unwrap() + term.value(fit.coefficients()).unwrap(),
    );
    let runtime = RuntimeMultivariateProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
        datafit,
        AxisPenalty::new(layout, PenaltyAxis::Responses, Box::new(EuclideanNorm(0.5))),
    )
    .normalize(Normalization::None)
    .fit_with(&solver)
    .unwrap();
    for (&actual, &expected) in runtime.coefficients().iter().zip(fit.coefficients()) {
        close(actual, expected);
    }
    close(runtime.objective(), fit.objective());
}

#[test]
fn separable_fits_match_independent_responses_including_normalization() {
    let x = Matrix::from_rows(&[
        &[0.0, 1.0],
        &[2.0, 0.0],
        &[4.0, 3.0],
        &[6.0, 1.0],
        &[1.0, 4.0],
    ]);
    let y = [
        1.0, 2.0, -1.0, 4.0, 2.0, -4.0, 3.0, 0.5, 1.0, 2.0, 7.0, 7.0, 7.0, 7.0, 7.0,
    ];
    let layout = CoefficientLayout::new(2, 3).unwrap();
    let solver = ProximalGradient::<Vec<f64>>::new()
        .tolerance(1e-10)
        .max_iterations(100_000);
    for intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Auto,
            Normalization::Standardize,
        ] {
            for penalty in [
                ElasticNet::new(0.15, 0.0),
                ElasticNet::new(0.0, 0.3),
                ElasticNet::new(0.15, 0.3),
            ] {
                let fit = Problem::new(
                    &x,
                    MultivariateGaussian::new(&y, 5, 3).fit_intercept(intercept),
                    AxisPenalty::new(layout, PenaltyAxis::Features, penalty),
                )
                .normalize(normalization)
                .fit_with(&solver)
                .unwrap();
                assert_eq!(fit.termination(), Termination::Converged);
                let predictions = fit.predict(&x).unwrap();
                let mut objective = 0.0;
                for (response, column) in y.chunks_exact(5).enumerate() {
                    let independent =
                        Problem::new(&x, Gaussian::new(column).fit_intercept(intercept), penalty)
                            .normalize(normalization)
                            .fit_with(&solver)
                            .unwrap();
                    assert_eq!(independent.termination(), Termination::Converged);
                    for (&actual, &expected) in fit
                        .coefficients_for_response(response)
                        .unwrap()
                        .iter()
                        .zip(independent.coefficients())
                    {
                        close(actual, expected);
                    }
                    close(fit.intercepts()[response], independent.intercept());
                    for (&actual, expected) in predictions[response * 5..(response + 1) * 5]
                        .iter()
                        .zip(independent.predict(&x).unwrap())
                    {
                        close(actual, expected);
                    }
                    objective += independent.objective();
                }
                close(fit.objective(), objective);
                assert!(fit.coefficients_for_response(3).is_none());
            }
        }
    }
}

#[test]
fn supplied_normalization_preserves_joint_predictions_and_penalty_scale() {
    let x = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let normalized = Matrix::from_rows(&[&[-1.5, -0.5], &[-0.5, -2.5], &[0.5, 3.5], &[1.5, -0.5]]);
    let y = [1.0, 2.0, -1.0, 4.0, -3.0, 2.0, 0.0, 1.0];
    let layout = CoefficientLayout::new(2, 2).unwrap();
    let solver = ProximalGradient::<Vec<f64>>::new()
        .tolerance(1e-10)
        .max_iterations(100_000);
    for intercept in [false, true] {
        for axis in [
            PenaltyAxis::All,
            PenaltyAxis::Features,
            PenaltyAxis::Responses,
        ] {
            let term = AxisPenalty::new(layout, axis, EuclideanNorm(0.2));
            let datafit = MultivariateGaussian::new(&y, 4, 2).fit_intercept(intercept);
            let lazy = Problem::new(&x, datafit, term.clone())
                .with_centers(vec![3.0, 1.25])
                .with_scales(vec![2.0, 0.5])
                .fit_with(&solver)
                .unwrap();
            let explicit = Problem::new(&normalized, datafit, term.clone())
                .normalize(Normalization::None)
                .fit_with(&solver)
                .unwrap();
            assert_eq!(lazy.termination(), Termination::Converged);
            assert_eq!(explicit.termination(), Termination::Converged);
            close(lazy.objective(), explicit.objective());
            for (&a, b) in lazy
                .predict(&x)
                .unwrap()
                .iter()
                .zip(explicit.predict(&normalized).unwrap())
            {
                close(a, b);
            }
            let beta: Vec<_> = lazy
                .coefficients()
                .iter()
                .enumerate()
                .map(|(i, b)| b * [2.0, 0.5][i % 2])
                .collect();
            close(
                lazy.objective(),
                datafit.value(&lazy.predict(&x).unwrap()).unwrap() + term.value(&beta).unwrap(),
            );
            assert!(lazy.preprocessing().centers_were_supplied());
            assert!(lazy.preprocessing().scales_were_supplied());
            if !intercept {
                for response in 0..2 {
                    close(
                        lazy.intercepts()[response],
                        -3.0 * lazy.coefficients()[2 * response]
                            - 1.25 * lazy.coefficients()[2 * response + 1],
                    );
                }
            }
        }
    }
}

#[test]
fn zero_features_and_single_response_keep_intercepts_and_shapes_explicit() {
    let x = Matrix::empty(3, 0);
    let y = [1.0, 2.0, 3.0, -3.0, 1.0, -1.0];
    let layout = CoefficientLayout::new(0, 2).unwrap();
    assert!(layout.is_empty());
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    for axis in [
        PenaltyAxis::All,
        PenaltyAxis::Features,
        PenaltyAxis::Responses,
    ] {
        let fit = Problem::new(
            &x,
            MultivariateGaussian::new(&y, 3, 2),
            AxisPenalty::new(layout, axis, L1::new(1e6)),
        )
        .fit_with(&solver)
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!(fit.coefficients().is_empty());
        close(fit.intercepts()[0], 2.0);
        close(fit.intercepts()[1], -1.0);
        assert!(fit.predict(&Matrix::empty(0, 0)).unwrap().is_empty());
    }
    let x = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0]]);
    let fit = Problem::new(
        &x,
        MultivariateGaussian::new(&y[..3], 3, 1),
        AxisPenalty::new(
            CoefficientLayout::new(1, 1).unwrap(),
            PenaltyAxis::Features,
            L1::new(0.0),
        ),
    )
    .fit_with(&solver)
    .unwrap();
    close(fit.coefficients()[0], 1.0);
    close(fit.intercepts()[0], 0.0);
}

#[test]
fn iteration_limits_report_the_joint_mapping_at_returned_parameters() {
    let x = Matrix::from_rows(&[&[-1.0], &[1.0]]);
    let y = [1.0, 5.0, -6.0, 2.0];
    let layout = CoefficientLayout::new(1, 2).unwrap();
    let fit = Problem::new(
        &x,
        MultivariateGaussian::new(&y, 2, 2),
        AxisPenalty::new(layout, PenaltyAxis::Features, Ridge::new(0.5)),
    )
    .normalize(Normalization::None)
    .fit_with(
        &ProximalGradient::<Vec<f64>>::new()
            .initial_step(0.25)
            .max_iterations(1),
    )
    .unwrap();
    assert_eq!(fit.termination(), Termination::IterationLimit);
    assert_eq!(fit.iterations(), 1);
    close(fit.coefficients()[0], 0.5 / 1.125);
    close(fit.coefficients()[1], 1.0 / 1.125);
    close(fit.intercepts()[0], 0.75);
    close(fit.intercepts()[1], -0.5);
    let mapping = ((1.5 * fit.coefficients()[1] - 4.0) / (1.0 + 0.5 * fit.step_size()))
        .abs()
        .max((fit.intercepts()[0] - 3.0).abs());
    close(fit.stopping_value(), mapping);
    close(
        fit.objective(),
        MultivariateGaussian::new(&y, 2, 2)
            .value(&fit.predict(&x).unwrap())
            .unwrap()
            + Ridge::new(0.5).value(fit.coefficients()).unwrap(),
    );
    assert_eq!(
        fit.stopping_criterion(),
        StoppingCriterion::proximal_gradient_mapping(1e-6)
    );
    close(fit.stopping_threshold(), 1e-6);
}

#[test]
fn dimensions_nonfinite_values_and_unsupported_criteria_are_rejected() {
    assert!(CoefficientLayout::new(2, 0).is_err());
    assert!(CoefficientLayout::new(usize::MAX, 2).is_err());
    let layout = CoefficientLayout::new(2, 3).unwrap();
    assert_eq!(layout.index(1, 2), Some(5));
    assert_eq!(layout.index(2, 0), None);
    assert_eq!(layout.index(0, 3), None);
    let x = Matrix::empty(2, 2);
    let solver = ProximalGradient::<Vec<f64>>::new();
    for datafit in [
        MultivariateGaussian::new(&[], 0, 2),
        MultivariateGaussian::new(&[], 2, 0),
        MultivariateGaussian::new(&[0.0; 4], 3, 2),
        MultivariateGaussian::new(&[0.0; 3], 2, 2),
        MultivariateGaussian::new(&[0.0; 4], 2, 2),
        MultivariateGaussian::new(&[0.0, f64::NAN, 0.0, 0.0], 2, 2),
        MultivariateGaussian::new(&[], usize::MAX, 2),
    ] {
        assert!(matches!(
            Problem::new(
                &x,
                datafit,
                AxisPenalty::new(layout, PenaltyAxis::Features, Ridge::new(0.1))
            )
            .fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
    let valid = MultivariateGaussian::new(&[0.0; 6], 2, 3);
    assert!(
        matches!(MultivariateGaussian::new(&[], usize::MAX, 2).value(&[]),
        Err(FitError::InvalidInput { message }) if message.contains("overflow"))
    );
    for wrong_layout in [
        CoefficientLayout::new(1, 3).unwrap(),
        CoefficientLayout::new(3, 2).unwrap(),
    ] {
        assert!(matches!(
            Problem::new(
                &x,
                valid,
                AxisPenalty::new(wrong_layout, PenaltyAxis::Features, Ridge::new(0.1))
            )
            .fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
    assert!(valid.value(&[0.0; 5]).is_err());
    assert!(valid.value(&[f64::INFINITY; 6]).is_err());
    assert!(valid.gradient(&[0.0; 6], &mut [0.0; 5]).is_err());
    let problem = Problem::new(
        &x,
        valid,
        AxisPenalty::new(layout, PenaltyAxis::Features, Ridge::new(0.1)),
    );
    assert!(matches!(
        problem.fit_with(&solver.terminate_on(StoppingCriterion::duality_gap(1e-6))),
        Err(FitError::UnsupportedCombination { .. })
    ));
    let fit = problem
        .fit_with(&ProximalGradient::<Vec<f64>>::new())
        .unwrap();
    assert!(fit.predict(&Matrix::empty(1, 1)).is_err());
    assert!(
        fit.predict(&Matrix::from_rows(&[&[f64::INFINITY, 0.0]]))
            .is_err()
    );
    for axis in [
        PenaltyAxis::All,
        PenaltyAxis::Features,
        PenaltyAxis::Responses,
    ] {
        let term = AxisPenalty::new(layout, axis, Ridge::new(0.1));
        assert!(term.value(&[0.0; 5]).is_err());
        assert!(term.value(&[f64::NAN; 6]).is_err());
        assert!(term.prox(&[0.0; 6], 1.0, &mut [0.0; 5]).is_err());
        for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(term.prox(&[0.0; 6], step, &mut [0.0; 6]).is_err());
        }
        for strength in [-1.0, f64::NAN, f64::INFINITY] {
            let term = AxisPenalty::new(layout, axis, Ridge::new(strength));
            assert!(term.value(&[0.0; 6]).is_err());
            assert!(term.prox(&[0.0; 6], 1.0, &mut [0.0; 6]).is_err());
        }
    }
}

#[test]
fn joint_ridge_has_one_unpenalized_intercept_per_response() {
    let x = Matrix::from_rows(&[&[-1.0, 0.0], &[1.0, 0.0], &[0.0, -1.0], &[0.0, 1.0]]);
    let y = [1.0, 5.0, 4.0, 2.0, -6.0, 2.0, -3.0, -1.0];
    let layout = CoefficientLayout::new(2, 2).unwrap();
    let fit = Problem::new(
        &x,
        MultivariateGaussian::new(&y, 4, 2),
        AxisPenalty::new(layout, PenaltyAxis::Features, Ridge::new(0.5)),
    )
    .normalize(Normalization::None)
    .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-10))
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(fit.layout(), layout);
    for (&actual, expected) in fit.coefficients().iter().zip([1.0, -0.5, 2.0, 0.5]) {
        close(actual, expected);
    }
    close(fit.intercepts()[0], 3.0);
    close(fit.intercepts()[1], -2.0);
    assert_eq!(layout.index(1, 0), Some(1));
    assert_eq!(layout.index(0, 1), Some(2));
    let predictions = fit.predict(&x).unwrap();
    for (&actual, expected) in predictions
        .iter()
        .zip([2.0, 4.0, 3.5, 2.5, -4.0, 0.0, -2.5, -1.5])
    {
        close(actual, expected);
    }
}
