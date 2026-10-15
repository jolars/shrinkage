//! Scaled and masked vector penalties on feature rows.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    AxisPenalty, CoefficientLayout, Datafit, ElasticNet, FitError, Gaussian, Masked, MatrixDesign,
    MultivariateGaussian, Normalization, Penalty, PenaltyAxis, Problem, ProximalGradient,
    ProximalPenalty, RestrictedProximalPenalty, RuntimeMultivariateProblem, Scaled, Termination,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-8 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

// Coupling responses makes incorrect row gathers observable even when the
// same scale and mask apply to every feature.
#[derive(Clone, Copy)]
struct SquaredSum;

impl Penalty for SquaredSum {
    type Error = FitError;

    fn value(&self, input: &[f64]) -> Result<f64, FitError> {
        Ok(0.5 * input.iter().sum::<f64>().powi(2))
    }
}

impl ProximalPenalty for SquaredSum {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.prox_restricted(input, &vec![false; input.len()], step, output)
    }
}

impl RestrictedProximalPenalty for SquaredSum {
    fn value_restricted(&self, input: &[f64], mask: &[bool]) -> Result<f64, FitError> {
        self.validate_restricted_input(input, mask)?;
        let sum: f64 = input
            .iter()
            .zip(mask)
            .filter(|(_, excluded)| !**excluded)
            .map(|(&value, _)| value)
            .sum();
        Ok(0.5 * sum * sum)
    }

    fn prox_restricted(
        &self,
        input: &[f64],
        mask: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        self.validate_restricted_input(input, mask)?;
        let sum: f64 = input
            .iter()
            .zip(mask)
            .filter(|(_, excluded)| !**excluded)
            .map(|(&value, _)| value)
            .sum();
        let count = mask.iter().filter(|&&excluded| !excluded).count();
        let correction = step * sum / (1.0 + step * count as f64);
        for ((&value, &excluded), target) in input.iter().zip(mask).zip(output) {
            *target = if excluded { value } else { value - correction };
        }
        Ok(())
    }
}

#[test]
fn scaled_masked_rows_match_independent_values_and_proximal_optimality() {
    let layout = CoefficientLayout::new(3, 4).unwrap();
    let rows = [
        [3.0_f64, -1.0, 4.0, 2.0],
        [-2.0, 5.0, 1.0, -3.0],
        [0.5, 2.0, -4.0, 6.0],
    ];
    let input = [
        3.0, -2.0, 0.5, -1.0, 5.0, 2.0, 4.0, 1.0, -4.0, 2.0, -3.0, 6.0,
    ];
    for mask in [
        vec![false; 4],
        vec![true; 4],
        vec![false, true, false, false],
    ] {
        for scale in [0.0, 0.25, 2.0] {
            let penalties: [Box<dyn ProximalPenalty>; 2] = [
                Box::new(Scaled::new(scale, Masked::new(mask.clone(), SquaredSum))),
                Box::new(Masked::new(
                    mask.clone(),
                    Scaled::new(
                        scale,
                        Box::new(SquaredSum) as Box<dyn RestrictedProximalPenalty>,
                    ),
                )),
            ];
            for inner in penalties {
                let penalty = AxisPenalty::new(layout, PenaltyAxis::Responses, inner);
                let expected: f64 = rows
                    .iter()
                    .map(|row| {
                        let sum: f64 = row
                            .iter()
                            .zip(&mask)
                            .filter(|(_, excluded)| !**excluded)
                            .map(|(&value, _)| value)
                            .sum();
                        scale * 0.5 * sum * sum
                    })
                    .sum();
                close(penalty.value(&input).unwrap(), expected);
                for step in [0.125, 1.0, 4.0] {
                    let mut output = [f64::NAN; 12];
                    penalty.prox(&input, step, &mut output).unwrap();
                    for (feature, row) in rows.iter().enumerate() {
                        let sum: f64 = (0..4)
                            .filter(|&response| !mask[response])
                            .map(|response| output[response * 3 + feature])
                            .sum();
                        for response in 0..4 {
                            let index = layout.index(feature, response).unwrap();
                            assert_eq!(input[index], row[response]);
                            if mask[response] {
                                assert_eq!(output[index], input[index]);
                            } else {
                                close((output[index] - input[index]) / step + scale * sum, 0.0);
                            }
                        }
                    }
                }
                // Evaluation after several prox calls must not depend on the
                // last row left in the reusable gather buffer.
                close(penalty.value(&input).unwrap(), expected);
            }
        }
    }
}

#[test]
fn row_compositions_match_analytic_fits_and_keep_intercepts_unpenalized() {
    let x = Matrix::from_rows(&[
        &[-1.0, -1.0, 1.0],
        &[-1.0, 1.0, -1.0],
        &[1.0, -1.0, -1.0],
        &[1.0, 1.0, 1.0],
    ]);
    let coefficients = [2.0_f64, -3.0, 0.1, -1.0, 0.0, 4.0];
    let y = [6.1, -0.1, 9.9, 4.1, 3.0, -5.0, -7.0, 1.0];
    let layout = CoefficientLayout::new(3, 2).unwrap();
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    for (axis, mask) in [
        (PenaltyAxis::Responses, vec![true, false]),
        (PenaltyAxis::Features, vec![true, false, true]),
    ] {
        for scale in [0.0, 0.5, 2.0] {
            for intercept in [false, true] {
                let datafit = MultivariateGaussian::new(&y, 4, 2).fit_intercept(intercept);
                let penalty = AxisPenalty::new(
                    layout,
                    axis,
                    Scaled::new(scale, Masked::new(mask.clone(), ElasticNet::new(0.2, 0.4))),
                );
                let typed = Problem::new(&x, datafit, penalty.clone())
                    .normalize(Normalization::None)
                    .fit_with(&solver)
                    .unwrap();
                let runtime = RuntimeMultivariateProblem::new(
                    Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
                    datafit,
                    AxisPenalty::new(
                        layout,
                        axis,
                        Box::new(penalty.penalty().clone()) as Box<dyn ProximalPenalty>,
                    ),
                )
                .normalize(Normalization::None)
                .fit_with(&solver)
                .unwrap();
                for fit in [&typed, &runtime] {
                    assert_eq!(fit.termination(), Termination::Converged);
                    assert!(fit.stopping_value() <= fit.stopping_threshold());
                    assert_eq!(fit.layout(), layout);
                    for response in 0..2 {
                        close(
                            fit.intercepts()[response],
                            if intercept {
                                [5.0, -2.0][response]
                            } else {
                                0.0
                            },
                        );
                        for feature in 0..3 {
                            let index = response * 3 + feature;
                            let coefficient = coefficients[index];
                            let excluded = mask[if axis == PenaltyAxis::Responses {
                                response
                            } else {
                                feature
                            }];
                            let expected = if excluded {
                                coefficient
                            } else {
                                coefficient.signum() * (coefficient.abs() - scale * 0.2).max(0.0)
                                    / (1.0 + scale * 0.4)
                            };
                            close(fit.coefficients()[index], expected);
                        }
                    }
                    close(
                        fit.objective(),
                        datafit.value(&fit.predict(&x).unwrap()).unwrap()
                            + penalty.value(fit.coefficients()).unwrap(),
                    );
                }
            }
        }
    }
}

#[test]
fn feature_row_masks_match_independent_normalized_response_fits() {
    let x = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let y = [1.0, 2.0, -1.0, 4.0, -3.0, 2.0, 0.0, 1.0, 7.0, 7.0, 7.0, 7.0];
    let layout = CoefficientLayout::new(2, 3).unwrap();
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    for normalization in [
        Normalization::None,
        Normalization::Auto,
        Normalization::Standardize,
    ] {
        for intercept in [false, true] {
            let inner = Scaled::new(
                0.5,
                Masked::new(vec![true, false], ElasticNet::new(0.2, 0.4)),
            );
            let fit = Problem::new(
                &x,
                MultivariateGaussian::new(&y, 4, 3).fit_intercept(intercept),
                AxisPenalty::new(layout, PenaltyAxis::Features, inner.clone()),
            )
            .normalize(normalization)
            .fit_with(&solver)
            .unwrap();
            assert_eq!(fit.termination(), Termination::Converged);
            let predictions = fit.predict(&x).unwrap();
            let mut objective = 0.0;
            for (response, targets) in y.chunks_exact(4).enumerate() {
                let independent = Problem::new(
                    &x,
                    Gaussian::new(targets).fit_intercept(intercept),
                    inner.clone(),
                )
                .normalize(normalization)
                .fit_with(&solver)
                .unwrap();
                assert_eq!(independent.termination(), Termination::Converged);
                objective += independent.objective();
                close(fit.intercepts()[response], independent.intercept());
                for (&actual, &expected) in fit
                    .coefficients_for_response(response)
                    .unwrap()
                    .iter()
                    .zip(independent.coefficients())
                {
                    close(actual, expected);
                }
                for (&actual, expected) in predictions[response * 4..(response + 1) * 4]
                    .iter()
                    .zip(independent.predict(&x).unwrap())
                {
                    close(actual, expected);
                }
            }
            close(fit.objective(), objective);
        }
    }
}

#[test]
fn empty_rows_and_zero_scales_still_validate_masks_and_parameters() {
    let x = Matrix::empty(3, 0);
    let y = [1.0, 2.0, 3.0, -3.0, 1.0, -1.0];
    let fit = Problem::new(
        &x,
        MultivariateGaussian::new(&y, 3, 2),
        AxisPenalty::new(
            CoefficientLayout::new(0, 2).unwrap(),
            PenaltyAxis::Responses,
            Scaled::new(
                0.5,
                Masked::new(vec![true, false], ElasticNet::new(0.2, 0.4)),
            ),
        ),
    )
    .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-10))
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert!(fit.coefficients().is_empty());
    close(fit.intercepts()[0], 2.0);
    close(fit.intercepts()[1], -1.0);

    for nfeatures in [0, 3] {
        let layout = CoefficientLayout::new(nfeatures, 2).unwrap();
        let input = vec![1.0; layout.len()];
        for (axis, width) in [
            (PenaltyAxis::Responses, 2),
            (PenaltyAxis::Features, nfeatures),
        ] {
            for (valid, mask, scale, inner) in [
                (true, vec![true; width], 0.0, ElasticNet::new(0.2, 0.4)),
                (false, vec![true; width + 1], 0.0, ElasticNet::new(0.2, 0.4)),
                (false, vec![true; width], -1.0, ElasticNet::new(0.2, 0.4)),
                (false, vec![true; width], 0.0, ElasticNet::new(-1.0, 0.4)),
            ] {
                let penalty =
                    AxisPenalty::new(layout, axis, Scaled::new(scale, Masked::new(mask, inner)));
                assert_eq!(penalty.validate_input(&input).is_ok(), valid);
                assert_eq!(penalty.value(&input).is_ok(), valid);
                let mut output = vec![f64::NAN; input.len()];
                assert_eq!(penalty.prox(&input, 1.0, &mut output).is_ok(), valid);
                if valid {
                    assert_eq!(output, input);
                    assert_eq!(penalty.value(&input).unwrap(), 0.0);
                    assert!(penalty.prox(&input, 0.0, &mut output).is_err());
                    assert!(penalty.value(&vec![0.0; layout.len() + 2]).is_err());
                    if !input.is_empty() {
                        assert!(penalty.value(&vec![f64::NAN; layout.len()]).is_err());
                    }
                }
            }
        }
    }
}
