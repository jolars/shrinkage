//! Explicit unpenalized coordinates and exact restricted proximal maps.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    Datafit, ElasticNet, FitError, Gaussian, L1, Masked, MatrixDesign, Normalization, Penalty,
    Problem, ProximalGradient, ProximalPenalty, RestrictedProximalPenalty, Ridge, RuntimeProblem,
    Scaled, Termination,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn restricted_values_and_maps_satisfy_coordinate_optimality() {
    let input = [-4.0, 2.0, -0.0, 0.1, -0.2, 6.0];
    for (l1, l2) in [(0.5, 0.0), (0.0, 0.75), (0.5, 0.75)] {
        for mask in [
            vec![false; 6],
            vec![true; 6],
            vec![true, false, true, false, false, true],
        ] {
            let inner: Box<dyn RestrictedProximalPenalty> = match (l1, l2) {
                (_, 0.0) => Box::new(L1::new(l1)),
                (0.0, _) => Box::new(Ridge::new(l2)),
                _ => Box::new(ElasticNet::new(l1, l2)),
            };
            let penalty = Masked::new(mask.clone(), inner);
            let zeroed: Vec<_> = input
                .iter()
                .zip(&mask)
                .map(|(&value, &excluded)| if excluded { 0.0 } else { value })
                .collect();
            close(
                penalty.value(&input).unwrap(),
                penalty.penalty().value(&zeroed).unwrap(),
            );
            assert_eq!(penalty.unpenalized(), mask);
            for step in [0.125, 1.0, 4.0] {
                let mut output = [f64::NAN; 6];
                penalty.prox(&input, step, &mut output).unwrap();
                for ((&value, &target), &excluded) in input.iter().zip(&output).zip(&mask) {
                    if excluded {
                        assert_eq!(target.to_bits(), value.to_bits());
                    } else {
                        let gradient = (target - value) / step + l2 * target;
                        if target == 0.0 {
                            assert!(gradient.abs() <= l1 + 1e-12);
                        } else {
                            close(gradient + l1 * target.signum(), 0.0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn excluded_coordinates_do_not_enter_penalty_arithmetic() {
    let input = [f64::MAX, -0.0, 2.0, -f64::MAX];
    let penalty = Masked::new(vec![true, true, false, true], ElasticNet::new(0.5, 0.75));
    close(penalty.penalty().l1(), 0.5);
    close(penalty.value(&input).unwrap(), 2.5);
    let mut output = [f64::NAN; 4];
    penalty.prox(&input, 1.0, &mut output).unwrap();
    close(output[2], 1.5 / 1.75);
    for j in [0, 1, 3] {
        assert_eq!(output[j].to_bits(), input[j].to_bits());
    }

    let penalty = Masked::new(vec![true; 4], ElasticNet::new(f64::MAX, f64::MAX));
    assert_eq!(penalty.value(&input).unwrap(), 0.0);
    penalty.prox(&input, f64::MAX, &mut output).unwrap();
    assert_eq!(output.map(f64::to_bits), input.map(f64::to_bits));

    let empty = Masked::new(vec![], Ridge::new(f64::MAX));
    assert_eq!(empty.value(&[]).unwrap(), 0.0);
    empty.prox(&[], f64::MAX, &mut []).unwrap();
}

#[test]
fn masks_steps_and_all_coordinates_are_validated() {
    let inner = ElasticNet::new(0.5, 0.75);
    for mask in [vec![], vec![true], vec![false; 3]] {
        let penalty = Masked::new(mask.clone(), inner);
        assert!(matches!(
            penalty.value(&[1.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            penalty.validate_input(&[1.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            penalty.prox(&[1.0; 2], 1.0, &mut [0.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            inner.value_restricted(&[1.0; 2], &mask),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            inner.prox_restricted(&[1.0; 2], &mask, 1.0, &mut [0.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
    }
    for mask in [vec![false; 2], vec![true; 2], vec![true, false]] {
        let penalty = Masked::new(mask, inner);
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
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for input in [[invalid, 1.0], [1.0, invalid]] {
                assert!(matches!(
                    penalty.value(&input),
                    Err(FitError::InvalidInput { .. })
                ));
                assert!(matches!(
                    penalty.prox(&input, 1.0, &mut [0.0; 2]),
                    Err(FitError::InvalidInput { .. })
                ));
            }
        }
    }
}

#[test]
fn excluded_coordinates_and_zero_scaling_do_not_hide_invalid_strengths() {
    for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for inner in [
            Box::new(L1::new(invalid)) as Box<dyn RestrictedProximalPenalty>,
            Box::new(Ridge::new(invalid)),
            Box::new(ElasticNet::new(invalid, 0.0)),
            Box::new(ElasticNet::new(0.0, invalid)),
            Box::new(ElasticNet::from_alpha(0.0, invalid)),
        ] {
            let penalty = Scaled::new(0.0, Masked::new(vec![true; 2], inner));
            assert!(matches!(
                penalty.value(&[1.0; 2]),
                Err(FitError::InvalidInput { .. })
            ));
            assert!(matches!(
                penalty.prox(&[1.0; 2], 1.0, &mut [0.0; 2]),
                Err(FitError::InvalidInput { .. })
            ));
            assert!(penalty.penalty().value(&[1.0; 2]).is_err());
            assert!(
                penalty
                    .penalty()
                    .prox(&[1.0; 2], 1.0, &mut [0.0; 2])
                    .is_err()
            );
        }
    }
}

#[test]
fn scaling_and_runtime_selection_preserve_restrictions() {
    let input = [-f64::MAX, 2.0, -4.0];
    for scale in [0.0, 0.25, 2.0] {
        let mask = vec![true, false, false];
        let penalties: [Box<dyn ProximalPenalty>; 3] = [
            Box::new(Scaled::new(
                scale,
                Masked::new(mask.clone(), ElasticNet::new(0.5, 0.75)),
            )),
            Box::new(Masked::new(
                mask.clone(),
                Scaled::new(scale, ElasticNet::new(0.5, 0.75)),
            )),
            Box::new(Masked::new(
                mask,
                ElasticNet::new(scale * 0.5, scale * 0.75),
            )),
        ];
        let expected = penalties[2].value(&input).unwrap();
        for penalty in &penalties {
            close(penalty.value(&input).unwrap(), expected);
            for step in [0.125, 1.0, 4.0] {
                let mut output = [f64::NAN; 3];
                let mut reference = [f64::NAN; 3];
                penalty.prox(&input, step, &mut output).unwrap();
                penalties[2].prox(&input, step, &mut reference).unwrap();
                assert_eq!(output[0].to_bits(), input[0].to_bits());
                for j in 1..3 {
                    close(output[j], reference[j]);
                }
            }
        }
    }
    let penalty = Scaled::new(0.0, Masked::new(vec![false, true], Ridge::new(f64::MAX)));
    assert_eq!(penalty.value(&[f64::MAX; 2]).unwrap(), 0.0);
    assert!(penalty.value(&[1.0]).is_err());
    let mut output = [0.0; 2];
    penalty.prox(&[f64::MAX; 2], 1.0, &mut output).unwrap();
    assert_eq!(output, [f64::MAX; 2]);

    let penalty = Masked::new(vec![false, true], Scaled::new(0.0, Ridge::new(f64::MAX)));
    assert_eq!(penalty.value(&[f64::MAX; 2]).unwrap(), 0.0);
    penalty.prox(&[f64::MAX; 2], f64::MAX, &mut output).unwrap();
    assert_eq!(output, [f64::MAX; 2]);
    for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let penalty = Masked::new(vec![true; 2], Scaled::new(invalid, Ridge::new(1.0)));
        assert!(matches!(
            penalty.value(&[1.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            penalty.prox(&[1.0; 2], 1.0, &mut output),
            Err(FitError::InvalidInput { .. })
        ));
    }
}

#[test]
fn retained_coordinates_report_nonfinite_arithmetic() {
    let mask = vec![true, false];
    assert!(matches!(
        Masked::new(mask.clone(), L1::new(f64::MAX)).value(&[1.0, 2.0]),
        Err(FitError::NumericalFailure { .. })
    ));
    assert!(matches!(
        Masked::new(mask.clone(), Ridge::new(f64::MAX)).value(&[1.0, 2.0]),
        Err(FitError::NumericalFailure { .. })
    ));
    assert!(matches!(
        Masked::new(mask.clone(), L1::new(f64::MAX)).prox(&[1.0; 2], 2.0, &mut [0.0; 2]),
        Err(FitError::NumericalFailure { .. })
    ));
    for (scale, step) in [(f64::MAX, 2.0), (f64::MIN_POSITIVE, f64::MIN_POSITIVE)] {
        assert!(matches!(
            Masked::new(mask.clone(), Scaled::new(scale, L1::new(1.0))).prox(
                &[1.0; 2],
                step,
                &mut [0.0; 2]
            ),
            Err(FitError::NumericalFailure { .. })
        ));
    }
}

// This coupled quadratic requires a different prox when a coordinate is excluded.
struct CoupledQuadratic;

impl Penalty for CoupledQuadratic {
    type Error = FitError;

    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        L1::new(0.0).validate_input(coefficients)?;
        Ok(0.5 * coefficients.iter().sum::<f64>().powi(2))
    }
}

impl ProximalPenalty for CoupledQuadratic {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.prox_restricted(input, &vec![false; input.len()], step, output)
    }
}

impl RestrictedProximalPenalty for CoupledQuadratic {
    fn value_restricted(
        &self,
        coefficients: &[f64],
        unpenalized: &[bool],
    ) -> Result<f64, FitError> {
        self.validate_restricted_input(coefficients, unpenalized)?;
        let sum: f64 = coefficients
            .iter()
            .zip(unpenalized)
            .filter(|(_, excluded)| !**excluded)
            .map(|(value, _)| value)
            .sum();
        Ok(0.5 * sum.powi(2))
    }

    fn prox_restricted(
        &self,
        input: &[f64],
        unpenalized: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        self.validate_restricted_input(input, unpenalized)?;
        if input.len() != output.len() || !step.is_finite() || step <= 0.0 {
            return Err(FitError::InvalidInput {
                message: "invalid quadratic proximal input".into(),
            });
        }
        let sum: f64 = input
            .iter()
            .zip(unpenalized)
            .filter(|(_, excluded)| !**excluded)
            .map(|(value, _)| value)
            .sum();
        let count = unpenalized.iter().filter(|&&excluded| !excluded).count();
        let correction = step * sum / (1.0 + step * count as f64);
        for ((&value, &excluded), target) in input.iter().zip(unpenalized).zip(output) {
            *target = if excluded { value } else { value - correction };
        }
        Ok(())
    }
}

#[test]
fn downstream_coupled_penalties_supply_their_own_restricted_problem() {
    let input = [10.0, 3.0, -1.0];
    let penalty = Masked::new(vec![true, false, false], CoupledQuadratic);
    close(penalty.value(&input).unwrap(), 2.0);
    let mut output = [f64::NAN; 3];
    penalty.prox(&input, 0.5, &mut output).unwrap();
    assert_eq!(output, [10.0, 2.5, -1.5]);
    for j in [1, 2] {
        close((output[j] - input[j]) / 0.5 + output[1] + output[2], 0.0);
    }
    let mut complete = [f64::NAN; 3];
    CoupledQuadratic.prox(&input, 0.5, &mut complete).unwrap();
    assert_ne!(output[1..], complete[1..]);
}

#[test]
fn masked_gaussian_fits_match_analytic_solutions() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [2.0, -4.0, 6.0, 0.0];
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    for (l1, l2) in [(0.5, 0.0), (0.0, 0.75), (0.5, 0.75)] {
        for mask in [
            vec![false; 2],
            vec![true; 2],
            vec![true, false],
            vec![false, true],
        ] {
            for intercept in [false, true] {
                let datafit = Gaussian::new(&y).fit_intercept(intercept);
                let penalty = Masked::new(mask.clone(), ElasticNet::new(l1, l2));
                let fit = Problem::new(&x, datafit, penalty.clone())
                    .normalize(Normalization::None)
                    .fit_with(&solver)
                    .unwrap();
                let runtime = RuntimeProblem::new(
                    Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
                    Box::new(datafit),
                    Box::new(penalty.clone()),
                )
                .normalize(Normalization::None)
                .fit_with(&solver)
                .unwrap();
                for actual in [&fit, &runtime] {
                    assert_eq!(actual.termination(), Termination::Converged);
                    assert!(actual.stopping_value() <= actual.stopping_threshold());
                    close(actual.intercept(), if intercept { 1.0 } else { 0.0 });
                    for ((&coefficient, correlation), &excluded) in
                        actual.coefficients().iter().zip([2.0_f64, -3.0]).zip(&mask)
                    {
                        let expected = if excluded {
                            correlation
                        } else {
                            correlation.signum() * (correlation.abs() - l1) / (1.0 + l2)
                        };
                        close(coefficient, expected);
                    }
                    let predictions = actual.predict(&x).unwrap();
                    let loss: f64 = predictions
                        .iter()
                        .zip(y)
                        .map(|(eta, target)| (eta - target).powi(2))
                        .sum::<f64>()
                        / 8.0;
                    close(
                        actual.objective(),
                        loss + penalty.value(actual.coefficients()).unwrap(),
                    );
                }
            }
        }
    }
}

#[test]
fn lazy_normalization_preserves_restricted_values_and_predictions() {
    let x = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let normalized = Matrix::from_rows(&[&[-1.5, -0.5], &[-0.5, -2.5], &[0.5, 3.5], &[1.5, -0.5]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    for intercept in [false, true] {
        for mask in [vec![true, false], vec![false, true]] {
            let datafit = Gaussian::new(&y).fit_intercept(intercept);
            let penalty = Masked::new(mask, ElasticNet::new(0.15, 0.3));
            let lazy = Problem::new(&x, datafit, penalty.clone())
                .with_centers(vec![3.0, 1.25])
                .with_scales(vec![2.0, 0.5])
                .fit_with(&solver)
                .unwrap();
            let explicit = Problem::new(&normalized, datafit, penalty.clone())
                .normalize(Normalization::None)
                .fit_with(&solver)
                .unwrap();
            assert_eq!(lazy.termination(), Termination::Converged);
            assert_eq!(explicit.termination(), Termination::Converged);
            close(lazy.objective(), explicit.objective());
            let theta: Vec<_> = lazy
                .coefficients()
                .iter()
                .zip([2.0, 0.5])
                .map(|(b, s)| b * s)
                .collect();
            for (&a, &b) in theta.iter().zip(explicit.coefficients()) {
                close(a, b);
            }
            let predictions = lazy.predict(&x).unwrap();
            close(
                lazy.objective(),
                datafit.value(&predictions).unwrap() + penalty.value(&theta).unwrap(),
            );
            for (a, b) in predictions
                .into_iter()
                .zip(explicit.predict(&normalized).unwrap())
            {
                close(a, b);
            }
            if !intercept {
                close(
                    lazy.intercept(),
                    -3.0 * lazy.coefficients()[0] - 1.25 * lazy.coefficients()[1],
                );
            }
        }
    }
}

#[test]
fn fits_validate_masks_and_parameters_without_features() {
    let x = Matrix::empty(3, 0);
    let y = [1.0, 2.0, 3.0];
    let solver = ProximalGradient::<Vec<f64>>::new();
    let fit = Problem::new(&x, Gaussian::new(&y), Masked::new(vec![], L1::new(1.0)))
        .fit_with(&solver)
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.intercept(), 2.0);
    for penalty in [
        Masked::new(vec![true], L1::new(1.0)),
        Masked::new(vec![], L1::new(-1.0)),
    ] {
        assert!(matches!(
            Problem::new(&x, Gaussian::new(&y), penalty).fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
}
