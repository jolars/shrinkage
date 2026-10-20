//! Weighted block optimality and independent Gaussian reference solutions.

#[path = "common/matrix.rs"]
mod matrix;

use shrinkage::{
    Datafit, DisjointGroups, FitError, Gaussian, GroupLasso, L1, MatrixDesign, Normalization,
    Penalty, Problem, ProximalGradient, ProximalPenalty, RuntimeProblem, Scaled, Termination,
    UncoveredPolicy,
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn penalty(lambda: f64) -> GroupLasso {
    GroupLasso::new(
        lambda,
        DisjointGroups::new(
            6,
            vec![vec![2, 0], vec![1], vec![4, 3]],
            vec![2.0, 0.5, 0.0],
            UncoveredPolicy::Unpenalized,
        )
        .unwrap(),
    )
}

#[test]
fn explicit_weights_and_uncovered_coordinates_define_the_complete_term() {
    let penalty = penalty(0.5);
    assert_eq!(penalty.lambda(), 0.5);
    assert_eq!(penalty.groups().groups(), [vec![2, 0], vec![1], vec![4, 3]]);
    assert_eq!(penalty.groups().weights(), [2.0, 0.5, 0.0]);
    assert_eq!(penalty.groups().uncovered(), [5]);
    let input = [3.0, -2.0, 4.0, -8.0, 6.0, 9.0];
    close(penalty.value(&input).unwrap(), 5.5, 1e-14);
    let mut output = [f64::NAN; 6];
    penalty.prox(&input, 2.0, &mut output).unwrap();
    for (actual, expected) in output.into_iter().zip([1.8, -1.5, 2.4, -8.0, 6.0, 9.0]) {
        close(actual, expected, 1e-14);
    }
    // Reusing output must overwrite both penalized and free coordinates.
    penalty.prox(&[0.0; 6], 2.0, &mut output).unwrap();
    assert_eq!(output, [0.0; 6]);
}

fn certify_prox(penalty: &GroupLasso, input: &[f64], output: &[f64], step: f64) {
    for (group, &weight) in penalty
        .groups()
        .groups()
        .iter()
        .zip(penalty.groups().weights())
    {
        let norm = group.iter().map(|&j| output[j].powi(2)).sum::<f64>().sqrt();
        let strength = penalty.lambda() * weight;
        if norm == 0.0 {
            let dual_norm = group
                .iter()
                .map(|&j| ((input[j] - output[j]) / step).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(dual_norm <= strength + 1e-12);
        } else {
            for &j in group {
                close(
                    (input[j] - output[j]) / step,
                    strength * output[j] / norm,
                    1e-12,
                );
            }
        }
    }
    for &j in penalty.groups().uncovered() {
        assert_eq!(output[j], input[j]);
    }
}

#[test]
fn block_subgradients_certify_proximal_optimality_including_zero_groups() {
    for lambda in [0.0, 0.1, 1.0, 10.0] {
        let penalty = penalty(lambda);
        for trial in 0..80 {
            let input: Vec<_> = (0..6)
                .map(|j| ((trial * 17 + j * 31) % 23) as f64 / 3.0 - 4.0)
                .collect();
            for step in [0.125, 1.0, 3.0] {
                let mut output = [f64::NAN; 6];
                penalty.prox(&input, step, &mut output).unwrap();
                certify_prox(&penalty, &input, &output, step);
            }
        }
        let mut output = [f64::NAN; 6];
        penalty.prox(&[0.0; 6], 1.0, &mut output).unwrap();
        assert_eq!(output, [0.0; 6]);
    }
    let penalty = GroupLasso::new(
        1.0,
        DisjointGroups::new(2, vec![vec![0, 1]], vec![5.0], UncoveredPolicy::Reject).unwrap(),
    );
    let mut output = [f64::NAN; 2];
    penalty.prox(&[3.0, -4.0], 1.0, &mut output).unwrap();
    assert_eq!(output, [0.0; 2]);
}

#[test]
fn singletons_match_l1_and_empty_collections_follow_coverage_policy() {
    let input = [-4.0, -0.1, 0.0, 0.1, 3.0];
    for lambda in [0.0, 0.3, 10.0] {
        let groups = DisjointGroups::new(
            5,
            (0..5).rev().map(|j| vec![j]).collect(),
            vec![1.0; 5],
            UncoveredPolicy::Reject,
        )
        .unwrap();
        let group = GroupLasso::new(lambda, groups);
        close(
            group.value(&input).unwrap(),
            L1::new(lambda).value(&input).unwrap(),
            1e-14,
        );
        let (mut actual, mut expected) = ([0.0; 5], [0.0; 5]);
        group.prox(&input, 0.7, &mut actual).unwrap();
        L1::new(lambda).prox(&input, 0.7, &mut expected).unwrap();
        for (a, b) in actual.into_iter().zip(expected) {
            close(a, b, 1e-14);
        }
    }
    for n in [0, 5] {
        let group = GroupLasso::new(
            2.0,
            DisjointGroups::new(n, vec![], vec![], UncoveredPolicy::Unpenalized).unwrap(),
        );
        let input = vec![f64::MAX; n];
        let mut output = vec![0.0; n];
        assert_eq!(group.value(&input).unwrap(), 0.0);
        group.prox(&input, 1.0, &mut output).unwrap();
        assert_eq!(output, input);
    }
}

#[test]
fn invalid_parameters_dimensions_and_inputs_are_rejected_even_when_unpenalized() {
    for lambda in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            penalty(lambda).value(&[0.0; 6]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            penalty(lambda).prox(&[0.0; 6], 1.0, &mut [0.0; 6]),
            Err(FitError::InvalidInput { .. })
        ));
    }
    for lambda in [0.0, 1.0] {
        let penalty = penalty(lambda);
        for input in [
            vec![0.0; 5],
            vec![0.0; 7],
            vec![f64::NAN; 6],
            vec![f64::INFINITY; 6],
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
        for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                penalty.prox(&[0.0; 6], step, &mut [0.0; 6]),
                Err(FitError::InvalidInput { .. })
            ));
        }
        assert!(penalty.prox(&[0.0; 6], 1.0, &mut [0.0; 5]).is_err());
    }
    assert!(Scaled::new(0.0, penalty(1.0)).value(&[0.0; 5]).is_err());
    assert!(Scaled::new(0.0, penalty(-1.0)).value(&[0.0; 6]).is_err());
}

#[test]
fn scaled_norms_avoid_square_overflow_and_underflow_and_skip_zero_weights() {
    let groups =
        DisjointGroups::new(2, vec![vec![0, 1]], vec![1.0], UncoveredPolicy::Reject).unwrap();
    for magnitude in [1e-300, 1e300] {
        let penalty = GroupLasso::new(magnitude, groups.clone());
        let mut output = [0.0; 2];
        penalty
            .prox(&[3.0 * magnitude, -4.0 * magnitude], 1.0, &mut output)
            .unwrap();
        close(output[0] / magnitude, 2.4, 1e-14);
        close(output[1] / magnitude, -3.2, 1e-14);
    }
    let small = GroupLasso::new(0.25, groups.clone());
    close(
        small.value(&[f64::MAX; 2]).unwrap() / f64::MAX,
        0.25 * 2.0_f64.sqrt(),
        1e-14,
    );
    let mut output = [0.0; 2];
    small.prox(&[f64::MAX; 2], 1.0, &mut output).unwrap();
    assert_eq!(output, [f64::MAX; 2]);
    assert_eq!(
        GroupLasso::new(0.0, groups.clone())
            .value(&[f64::MAX; 2])
            .unwrap(),
        0.0
    );
    assert!(matches!(
        GroupLasso::new(2.0, groups.clone()).value(&[f64::MAX; 2]),
        Err(FitError::NumericalFailure { .. })
    ));
    assert!(matches!(
        GroupLasso::new(f64::MAX, groups).prox(&[1.0; 2], 2.0, &mut output),
        Err(FitError::NumericalFailure { .. })
    ));
    assert_eq!(
        penalty(1.0)
            .value(&[0.0, 0.0, 0.0, f64::MAX, f64::MAX, f64::MAX])
            .unwrap(),
        0.0
    );
}

#[test]
fn correlated_gaussian_fits_match_a_constructed_kkt_reference() {
    // Walsh columns are orthogonal with squared norm n. Mixing them gives a
    // correlated full-rank design whose residual has prescribed correlations.
    let signs: Vec<[f64; 5]> = (0..8_usize)
        .map(|i| {
            [1, 2, 3, 4, 5].map(|mask| {
                if (i & mask).count_ones() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
        })
        .collect();
    let rows: Vec<_> = signs
        .iter()
        .map(|h| {
            [
                h[0],
                h[1] + 0.5 * h[0],
                h[2] + 0.25 * h[0],
                h[3],
                h[4] + 0.5 * h[2],
            ]
        })
        .collect();
    let x = matrix::Matrix::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>());
    let groups = DisjointGroups::new(
        5,
        vec![vec![2, 0], vec![1, 3]],
        vec![2.0, 1.0],
        UncoveredPolicy::Unpenalized,
    )
    .unwrap();
    let penalty = GroupLasso::new(0.5, groups);
    for intercept in [0.0, 1.7] {
        let y: Vec<_> = rows
            .iter()
            .zip(&signs)
            .map(|(x, h)| {
                3.0 * x[0] - 4.0 * x[2] + 2.0 * x[4] + intercept + 0.6 * h[0]
                    - 0.2 * h[1]
                    - 0.95 * h[2]
                    - 0.2 * h[3]
                    + 0.475 * h[4]
            })
            .collect();
        let datafit = Gaussian::new(&y).fit_intercept(intercept != 0.0);
        let solver = ProximalGradient::<Vec<f64>>::new()
            .tolerance(1e-10)
            .initial_step(8.0);
        let typed = Problem::new(&x, datafit, penalty.clone())
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
        for fit in [typed, runtime] {
            assert_eq!(fit.termination(), Termination::Converged);
            close(fit.intercept(), intercept, 1e-9);
            for (&actual, expected) in fit.coefficients().iter().zip([3.0, 0.0, -4.0, 0.0, 2.0]) {
                close(actual, expected, 1e-9);
            }
            close(fit.objective(), 5.7840625, 1e-10);
            let predictions = fit.predict(&x).unwrap();
            close(
                fit.objective(),
                datafit.value(&predictions).unwrap() + penalty.value(fit.coefficients()).unwrap(),
                1e-12,
            );
            let correlations: Vec<_> = (0..5)
                .map(|j| {
                    rows.iter()
                        .zip(predictions.iter().zip(&y))
                        .map(|(row, (pred, y))| row[j] * (y - pred))
                        .sum::<f64>()
                        / 8.0
                })
                .collect();
            // These independently evaluated KKT equations include the inactive
            // block and the free coefficient, as well as the intercept.
            for (actual, expected) in correlations.into_iter().zip([0.6, 0.1, -0.8, -0.2, 0.0]) {
                close(actual, expected, 1e-9);
            }
        }
    }
}

#[test]
fn normalization_preserves_predictions_and_penalizes_optimization_coefficients() {
    let x = matrix::Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let normalized =
        matrix::Matrix::from_rows(&[&[-1.5, -0.5], &[-0.5, -2.5], &[0.5, 3.5], &[1.5, -0.5]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let groups =
        DisjointGroups::new(2, vec![vec![1, 0]], vec![1.3], UncoveredPolicy::Reject).unwrap();
    let penalty = GroupLasso::new(0.15, groups);
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    for fit_intercept in [false, true] {
        let datafit = Gaussian::new(&y).fit_intercept(fit_intercept);
        let lazy = Problem::new(&x, datafit, penalty.clone())
            .with_centers(vec![3.0, 1.25])
            .with_scales(vec![2.0, 0.5])
            .fit_with(&solver)
            .unwrap();
        let explicit = Problem::new(&normalized, datafit, penalty.clone())
            .normalize(Normalization::None)
            .fit_with(&solver)
            .unwrap();
        for fit in [&lazy, &explicit] {
            assert_eq!(fit.termination(), Termination::Converged);
        }
        close(lazy.objective(), explicit.objective(), 1e-12);
        for (&actual, expected) in lazy
            .predict(&x)
            .unwrap()
            .iter()
            .zip(explicit.predict(&normalized).unwrap())
        {
            close(actual, expected, 1e-10);
        }
        let theta: Vec<_> = lazy
            .coefficients()
            .iter()
            .zip([2.0, 0.5])
            .map(|(b, s)| b * s)
            .collect();
        for (&actual, &expected) in theta.iter().zip(explicit.coefficients()) {
            close(actual, expected, 1e-10);
        }
        close(
            lazy.objective(),
            datafit.value(&lazy.predict(&x).unwrap()).unwrap() + penalty.value(&theta).unwrap(),
            1e-12,
        );
        close(
            lazy.intercept(),
            explicit.intercept() - 3.0 * lazy.coefficients()[0] - 1.25 * lazy.coefficients()[1],
            1e-12,
        );
    }
}

#[test]
fn zero_design_and_empty_features_fit_the_free_intercept() {
    for p in [0, 3] {
        let x = matrix::Matrix::empty(3, p);
        let groups = DisjointGroups::new(
            p,
            (0..p).map(|j| vec![j]).collect(),
            vec![1.0; p],
            UncoveredPolicy::Reject,
        )
        .unwrap();
        let fit = Problem::new(
            &x,
            Gaussian::new(&[1.0, 2.0, 3.0]),
            GroupLasso::new(0.5, groups),
        )
        .fit_with(&ProximalGradient::<Vec<f64>>::new())
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert_eq!(fit.coefficients(), vec![0.0; p]);
        close(fit.intercept(), 2.0, 1e-12);
    }
    let x = matrix::Matrix::empty(3, 2);
    assert!(matches!(
        Problem::new(&x, Gaussian::new(&[1.0, 2.0, 3.0]), penalty(0.0))
            .fit_with(&ProximalGradient::<Vec<f64>>::new()),
        Err(FitError::InvalidInput { .. })
    ));
}
