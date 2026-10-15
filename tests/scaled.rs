//! Scaling complete penalties without changing their proximal capabilities.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    AxisPenalty, CoefficientLayout, ElasticNet, ErasedPathPenalty, FitError, Gaussian, L1,
    MatrixDesign, Normalization, Penalty, PenaltyAxis, Problem, ProximalGradient, ProximalPenalty,
    Ridge, RuntimeProblem, Scaled, Slope, Termination,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn same_penalty(actual: &dyn ProximalPenalty, expected: &dyn ProximalPenalty) {
    let input = [-4.0, 0.0, 1.5, -1.5, 6.0];
    close(
        actual.value(&input).unwrap(),
        expected.value(&input).unwrap(),
    );
    for step in [0.125, 1.0, 4.0] {
        let mut output = [f64::NAN; 5];
        let mut reference = [f64::NAN; 5];
        actual.prox(&input, step, &mut output).unwrap();
        expected.prox(&input, step, &mut reference).unwrap();
        for (a, b) in output.into_iter().zip(reference) {
            close(a, b);
        }
    }
}

#[test]
fn scaling_matches_all_builtin_strengths() {
    for scale in [0.0, 0.25, 1.0, 3.0] {
        same_penalty(&Scaled::new(scale, L1::new(0.5)), &L1::new(scale * 0.5));
        same_penalty(
            &Scaled::new(scale, Ridge::new(0.75)),
            &Ridge::new(scale * 0.75),
        );
        same_penalty(
            &Scaled::new(scale, ElasticNet::new(0.5, 0.75)),
            &ElasticNet::new(scale * 0.5, scale * 0.75),
        );
        let weights = vec![2.0, 1.25, 0.75, 0.25, 0.0];
        same_penalty(
            &Scaled::new(scale, Slope::new(weights.clone())),
            &Slope::new(weights.iter().map(|weight| scale * weight).collect()),
        );
    }
}

struct ValueOnly;

impl Penalty for ValueOnly {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        Ok(1.0 + coefficients.iter().map(|value| value.abs()).sum::<f64>())
    }
}

#[test]
fn value_only_boxed_and_nested_penalties_keep_their_semantics() {
    let penalty = Scaled::new(0.25, ValueOnly);
    assert_eq!(penalty.scale(), 0.25);
    assert_eq!(penalty.value(&[-2.0, 3.0]).unwrap(), 1.5);
    assert_eq!(penalty.value(&[]).unwrap(), 0.25);
    assert_eq!(Scaled::new(0.0, ValueOnly).value(&[]).unwrap(), 0.0);

    let inner: Box<dyn ProximalPenalty> = Box::new(ElasticNet::new(0.5, 0.75));
    let penalty = Scaled::new(0.25, inner);
    close(penalty.penalty().value(&[2.0]).unwrap(), 2.5);
    same_penalty(&penalty, &ElasticNet::new(0.125, 0.1875));
    same_penalty(
        &Scaled::new(3.0, Scaled::new(0.25, L1::new(0.5))),
        &L1::new(0.375),
    );
}

#[test]
fn zero_scale_is_zero_and_identity_even_when_the_inner_value_would_overflow() {
    let input = [f64::MAX, -f64::MAX];
    for scale in [0.0, -0.0] {
        let penalties: Vec<Box<dyn ProximalPenalty>> = vec![
            Box::new(Scaled::new(scale, L1::new(f64::MAX))),
            Box::new(Scaled::new(scale, Ridge::new(f64::MAX))),
            Box::new(Scaled::new(scale, ElasticNet::new(f64::MAX, f64::MAX))),
            Box::new(Scaled::new(scale, Slope::new(vec![f64::MAX; 2]))),
            Box::new(Scaled::new(
                scale,
                Box::new(Ridge::new(f64::MAX)) as Box<dyn ProximalPenalty>,
            )),
            Box::new(Scaled::new(scale, Scaled::new(f64::MAX, L1::new(f64::MAX)))),
        ];
        for penalty in penalties {
            assert_eq!(penalty.value(&input).unwrap(), 0.0);
            for step in [f64::MIN_POSITIVE, 1.0, f64::MAX] {
                let mut output = [f64::NAN; 2];
                penalty.prox(&input, step, &mut output).unwrap();
                assert_eq!(output, input);
            }
        }
        let empty = Scaled::new(scale, Slope::new(vec![]));
        assert_eq!(empty.value(&[]).unwrap(), 0.0);
        empty.prox(&[], 1.0, &mut []).unwrap();
    }
}

#[test]
fn zero_scale_preserves_validation_through_existing_wrappers() {
    let layout = CoefficientLayout::new(2, 2).unwrap();
    for axis in [
        PenaltyAxis::All,
        PenaltyAxis::Features,
        PenaltyAxis::Responses,
    ] {
        let penalty = Scaled::new(0.0, AxisPenalty::new(layout, axis, Ridge::new(f64::MAX)));
        let input = [f64::MAX; 4];
        assert_eq!(penalty.value(&input).unwrap(), 0.0);
        let mut output = [f64::NAN; 4];
        penalty.prox(&input, 1.0, &mut output).unwrap();
        assert_eq!(output, input);
        assert!(matches!(
            penalty.value(&[1.0]),
            Err(FitError::InvalidInput { .. })
        ));
        let invalid = Scaled::new(0.0, AxisPenalty::new(layout, axis, Ridge::new(-1.0)));
        assert!(matches!(
            invalid.value(&input),
            Err(FitError::InvalidInput { .. })
        ));

        let empty = CoefficientLayout::new(0, 2).unwrap();
        let width = if axis == PenaltyAxis::Responses { 2 } else { 0 };
        let penalty = Scaled::new(
            0.0,
            AxisPenalty::new(empty, axis, Slope::new(vec![1.0; width])),
        );
        assert_eq!(penalty.value(&[]).unwrap(), 0.0);
        penalty.prox(&[], 1.0, &mut []).unwrap();
        let invalid = Scaled::new(
            0.0,
            AxisPenalty::new(empty, axis, Slope::new(vec![1.0; width + 1])),
        );
        assert!(matches!(
            invalid.value(&[]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            invalid.prox(&[], 1.0, &mut []),
            Err(FitError::InvalidInput { .. })
        ));
    }
    let penalty = Scaled::new(0.0, ErasedPathPenalty::new(Ridge::new(f64::MAX)));
    assert_eq!(penalty.value(&[f64::MAX]).unwrap(), 0.0);
    let mut output = [0.0];
    penalty.prox(&[f64::MAX], 1.0, &mut output).unwrap();
    assert_eq!(output, [f64::MAX]);
}

#[test]
fn invalid_scales_and_inner_strengths_are_rejected() {
    for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let penalty = Scaled::new(invalid, L1::new(0.0));
        assert!(matches!(
            penalty.value(&[]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            penalty.prox(&[], 1.0, &mut []),
            Err(FitError::InvalidInput { .. })
        ));
        for scale in [0.0, 0.25] {
            let penalties: Vec<Box<dyn ProximalPenalty>> = vec![
                Box::new(Scaled::new(scale, L1::new(invalid))),
                Box::new(Scaled::new(scale, Ridge::new(invalid))),
                Box::new(Scaled::new(scale, ElasticNet::new(invalid, 0.0))),
                Box::new(Scaled::new(scale, ElasticNet::new(0.0, invalid))),
                Box::new(Scaled::new(scale, Slope::new(vec![invalid; 2]))),
                Box::new(Scaled::new(scale, Scaled::new(invalid, L1::new(1.0)))),
            ];
            for penalty in penalties {
                assert!(matches!(
                    penalty.value(&[1.0; 2]),
                    Err(FitError::InvalidInput { .. })
                ));
                assert!(matches!(
                    penalty.prox(&[1.0; 2], 1.0, &mut [0.0; 2]),
                    Err(FitError::InvalidInput { .. })
                ));
            }
        }
    }
}

#[test]
fn zero_scale_still_validates_inputs_dimensions_and_steps() {
    for scale in [0.0, 0.25] {
        let penalty = Scaled::new(scale, Slope::new(vec![2.0, 1.0]));
        for input in [
            vec![],
            vec![1.0],
            vec![1.0; 3],
            vec![f64::NAN, 1.0],
            vec![1.0, f64::INFINITY],
        ] {
            assert!(matches!(
                penalty.value(&input),
                Err(FitError::InvalidInput { .. })
            ));
            assert!(matches!(
                penalty.prox(&input, 1.0, &mut vec![0.0; input.len()]),
                Err(FitError::InvalidInput { .. })
            ));
        }
        for step in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                penalty.prox(&[1.0; 2], step, &mut [0.0; 2]),
                Err(FitError::InvalidInput { .. })
            ));
        }
        assert!(matches!(
            penalty.prox(&[1.0; 2], 1.0, &mut [0.0]),
            Err(FitError::InvalidInput { .. })
        ));
        let invalid = Scaled::new(scale, Slope::new(vec![1.0, 2.0]));
        assert!(matches!(
            invalid.value(&[1.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            invalid.prox(&[1.0; 2], 1.0, &mut [0.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
    }
    assert!(Scaled::new(0.0, ValueOnly).value(&[f64::NAN]).is_err());
}

#[test]
fn nonrepresentable_scaled_values_and_steps_report_numerical_failure() {
    assert!(matches!(
        Scaled::new(f64::MAX, L1::new(1.0)).value(&[2.0]),
        Err(FitError::NumericalFailure { .. })
    ));
    for (scale, step) in [(f64::MAX, 2.0), (f64::MIN_POSITIVE, f64::MIN_POSITIVE)] {
        assert!(matches!(
            Scaled::new(scale, L1::new(1.0)).prox(&[1.0], step, &mut [0.0]),
            Err(FitError::NumericalFailure { .. })
        ));
    }
    assert!(matches!(
        Scaled::new(1.0, L1::new(f64::MAX)).value(&[2.0]),
        Err(FitError::NumericalFailure { .. })
    ));
}

#[test]
fn scaled_fits_match_builtin_strengths_with_an_unpenalized_intercept() {
    let x = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    for scale in [0.0, 0.25, 2.0] {
        for intercept in [false, true] {
            for normalization in [Normalization::None, Normalization::Auto] {
                let datafit = Gaussian::new(&y).fit_intercept(intercept);
                let fit = Problem::new(&x, datafit, Scaled::new(scale, ElasticNet::new(0.2, 0.4)))
                    .normalize(normalization)
                    .fit_with(&solver)
                    .unwrap();
                let expected = Problem::new(&x, datafit, ElasticNet::new(scale * 0.2, scale * 0.4))
                    .normalize(normalization)
                    .fit_with(&solver)
                    .unwrap();
                let runtime = RuntimeProblem::new(
                    Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
                    Box::new(datafit),
                    Box::new(Scaled::new(scale, ElasticNet::new(0.2, 0.4))),
                )
                .normalize(normalization)
                .fit_with(&solver)
                .unwrap();
                for actual in [&fit, &runtime] {
                    assert_eq!(actual.termination(), Termination::Converged);
                    close(actual.objective(), expected.objective());
                    close(actual.intercept(), expected.intercept());
                    for (&a, &b) in actual.coefficients().iter().zip(expected.coefficients()) {
                        close(a, b);
                    }
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
    }
}

#[test]
fn scaled_fits_validate_penalties_without_features() {
    let x = Matrix::empty(3, 0);
    let y = [1.0, 2.0, 3.0];
    let solver = ProximalGradient::<Vec<f64>>::new();
    for scale in [0.0, 0.25] {
        let fit = Problem::new(&x, Gaussian::new(&y), Scaled::new(scale, L1::new(1.0)))
            .fit_with(&solver)
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        close(fit.intercept(), 2.0);
        assert!(fit.coefficients().is_empty());
        assert!(matches!(
            Problem::new(&x, Gaussian::new(&y), Scaled::new(scale, L1::new(-1.0)))
                .fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
}
