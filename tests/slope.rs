//! Exact sorted L1 proximal maps and independent optimality certificates.

#[path = "common/matrix.rs"]
mod matrix;

use shrinkage::{
    AxisPenalty, CoefficientLayout, Datafit, FitError, Gaussian, L1, MatrixDesign, Normalization,
    Penalty, PenaltyAxis, Problem, ProximalGradient, ProximalPenalty, RuntimeProblem, Slope,
    Termination,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn sorted_weights_define_the_complete_penalty_and_step() {
    let penalty = Slope::new(vec![3.0, 2.0, 0.5]);
    assert_eq!(penalty.weights(), &[3.0, 2.0, 0.5]);
    close(penalty.value(&[-1.0, 4.0, -2.0]).unwrap(), 16.5);
    let mut output = [f64::NAN; 3];
    penalty.prox(&[-1.0, 4.0, -2.0], 0.5, &mut output).unwrap();
    assert_eq!(output, [-0.75, 2.5, -1.0]);
}

#[test]
fn pooling_handles_ties_cascading_merges_and_a_zero_tail() {
    let cases = [
        (vec![5.0, 2.0], vec![4.0, 2.0], vec![1.0, 0.0]),
        (vec![3.0, -3.0], vec![3.0, 3.0], vec![0.0, 0.0]),
        (vec![2.0, 1.0], vec![3.0, 0.0], vec![0.0, 0.0]),
        (vec![-3.0, 3.0], vec![3.0, 1.0], vec![-1.0, 1.0]),
        (
            vec![4.0, -5.0, 1.0, 6.0],
            vec![6.0, 4.0, 0.0, 0.0],
            vec![5.0 / 3.0, -5.0 / 3.0, 1.0, 5.0 / 3.0],
        ),
        (
            vec![0.0, -0.0, 3.0, -3.0],
            vec![4.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, -0.5],
        ),
    ];
    for (input, weights, expected) in cases {
        let mut output = vec![f64::NAN; input.len()];
        Slope::new(weights).prox(&input, 1.0, &mut output).unwrap();
        for (&actual, expected) in output.iter().zip(expected) {
            close(actual, expected);
        }
    }
}

#[test]
fn constant_weights_match_l1_and_zero_weights_preserve_input() {
    let input = [-4.0, 1.0, -1.0, 0.0, 3.0];
    for weight in [0.0, 0.2, 2.0, 20.0] {
        let penalty = Slope::new(vec![weight; input.len()]);
        close(
            penalty.value(&input).unwrap(),
            L1::new(weight).value(&input).unwrap(),
        );
        for step in [0.1, 1.0, 5.0] {
            let mut actual = [0.0; 5];
            let mut expected = [0.0; 5];
            penalty.prox(&input, step, &mut actual).unwrap();
            L1::new(weight).prox(&input, step, &mut expected).unwrap();
            for (&actual, expected) in actual.iter().zip(expected) {
                close(actual, expected);
            }
        }
    }
    let empty = Slope::new(vec![]);
    assert_eq!(empty.value(&[]).unwrap(), 0.0);
    empty.prox(&[], 1.0, &mut []).unwrap();
    let extremes = [f64::MAX, -f64::MAX, f64::MAX];
    let mut output = [0.0; 3];
    let zero = Slope::new(vec![0.0; 3]);
    assert_eq!(zero.value(&extremes).unwrap(), 0.0);
    zero.prox(&extremes, 1.0, &mut output).unwrap();
    assert_eq!(output, extremes);
}

#[test]
fn invalid_weights_dimensions_steps_and_inputs_are_rejected() {
    for weights in [
        vec![-1.0, 0.0],
        vec![1.0, -0.1],
        vec![1.0, 2.0],
        vec![f64::NAN, 1.0],
        vec![1.0, f64::NAN],
        vec![f64::INFINITY, 1.0],
        vec![1.0, f64::NEG_INFINITY],
    ] {
        let penalty = Slope::new(weights);
        assert!(matches!(
            penalty.value(&[1.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(matches!(
            penalty.prox(&[1.0; 2], 1.0, &mut [0.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
    }
    let penalty = Slope::new(vec![2.0, 1.0]);
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
    assert!(matches!(
        penalty.prox(&[1.0; 2], 1.0, &mut [0.0]),
        Err(FitError::InvalidInput { .. })
    ));
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            penalty.prox(&[1.0; 2], step, &mut [0.0; 2]),
            Err(FitError::InvalidInput { .. })
        ));
        assert!(Slope::new(vec![]).prox(&[], step, &mut []).is_err());
    }
}

#[test]
fn nonfinite_arithmetic_is_reported_and_block_means_avoid_overflow() {
    let penalty = Slope::new(vec![f64::MAX, 0.0]);
    assert!(matches!(
        penalty.value(&[2.0, 0.0]),
        Err(FitError::NumericalFailure { .. })
    ));
    assert!(matches!(
        penalty.prox(&[1.0, 0.0], 2.0, &mut [0.0; 2]),
        Err(FitError::NumericalFailure { .. })
    ));
    assert!(matches!(
        Slope::new(vec![1.0; 2]).value(&[f64::MAX; 2]),
        Err(FitError::NumericalFailure { .. })
    ));
    let mut output = [0.0; 3];
    Slope::new(vec![f64::MAX, 0.0, 0.0])
        .prox(&[f64::MAX; 3], 1.0, &mut output)
        .unwrap();
    for actual in output {
        close(actual / f64::MAX, 2.0 / 3.0);
    }
}

// Every isotonic solution is constant on contiguous blocks. Enumerating all
// partitions gives a small independent oracle without the pooling algorithm.
fn exhaustive_prox(input: &[f64], weights: &[f64], step: f64) -> Vec<f64> {
    let mut order: Vec<_> = (0..input.len()).collect();
    order.sort_unstable_by(|&a, &b| input[b].abs().total_cmp(&input[a].abs()));
    let sorted: Vec<_> = order.iter().map(|&i| input[i].abs()).collect();
    let mut best = vec![0.0; input.len()];
    let mut best_objective = f64::INFINITY;
    for mask in 0..(1 << (input.len() - 1)) {
        let mut candidate = vec![0.0; input.len()];
        let mut start = 0;
        for end in 1..=input.len() {
            if end == input.len() || mask & (1 << (end - 1)) != 0 {
                let mean = (start..end)
                    .map(|i| sorted[i] - step * weights[i])
                    .sum::<f64>()
                    / (end - start) as f64;
                candidate[start..end].fill(mean.max(0.0));
                start = end;
            }
        }
        if candidate.windows(2).any(|pair| pair[0] < pair[1]) {
            continue;
        }
        let objective: f64 = (0..input.len())
            .map(|i| 0.5 * (candidate[i] - sorted[i]).powi(2) + step * weights[i] * candidate[i])
            .sum();
        if objective < best_objective {
            best_objective = objective;
            for (rank, &i) in order.iter().enumerate() {
                best[i] = candidate[rank].copysign(input[i]);
            }
        }
    }
    best
}

fn certify_optimality(input: &[f64], output: &[f64], penalty: &Slope, step: f64) {
    let subgradient: Vec<_> = input
        .iter()
        .zip(output)
        .map(|(y, z)| (y - z) / step)
        .collect();
    let mut sorted: Vec<_> = subgradient.iter().map(|g| g.abs()).collect();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let (mut gradient_sum, mut weight_sum) = (0.0, 0.0);
    // These prefix inequalities place the subgradient in the dual unit ball,
    // including when some or all weights vanish. Equality in the support
    // inequality then certifies (input - output) / step in the subdifferential.
    for (g, weight) in sorted.iter().zip(penalty.weights()) {
        gradient_sum += g;
        weight_sum += weight;
        assert!(
            gradient_sum <= weight_sum + 1e-9,
            "{gradient_sum} > {weight_sum}"
        );
    }
    close(
        output.iter().zip(&subgradient).map(|(z, g)| z * g).sum(),
        penalty.value(output).unwrap(),
    );
}

#[test]
fn exhaustive_partitions_and_subgradients_certify_optimality() {
    let mut state = 17_u64;
    let mut draw = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (state >> 32) as usize
    };
    for length in 1..=6 {
        for trial in 0..100 {
            let input: Vec<_> = (0..length)
                .map(|_| (draw() % 17) as f64 / 2.0 - 4.0)
                .collect();
            let mut weights: Vec<_> = (0..length).map(|_| (draw() % 9) as f64 / 4.0).collect();
            weights.sort_unstable_by(|a, b| b.total_cmp(a));
            if trial % 10 == 0 {
                weights.fill(0.0);
            }
            let penalty = Slope::new(weights);
            for step in [0.125, 1.0, 3.0] {
                let expected = exhaustive_prox(&input, penalty.weights(), step);
                let mut output = vec![f64::NAN; length];
                penalty.prox(&input, step, &mut output).unwrap();
                for (&actual, expected) in output.iter().zip(expected) {
                    close(actual, expected);
                }
                certify_optimality(&input, &output, &penalty, step);
            }
        }
    }
}

#[test]
fn gaussian_typed_and_runtime_fits_match_an_orthogonal_solution() {
    let x = matrix::Matrix::from_rows(&[
        &[-1.0, -1.0, 1.0],
        &[-1.0, 1.0, -1.0],
        &[1.0, -1.0, -1.0],
        &[1.0, 1.0, 1.0],
    ]);
    let y = [3.0, 3.0, -7.0, 9.0];
    let datafit = Gaussian::new(&y);
    let weights = vec![3.0, 0.5, 0.0];
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    let typed = Problem::new(&x, datafit, Slope::new(weights.clone()))
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
    let runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
        Box::new(datafit),
        Box::new(Slope::new(weights.clone())),
    )
    .normalize(Normalization::None)
    .fit_with(&solver)
    .unwrap();
    // X'X / n is the identity, X'y / n = [-1, 4, 4], and mean(y) = 2.
    for fit in [typed, runtime] {
        assert_eq!(fit.termination(), Termination::Converged);
        for (&actual, expected) in fit.coefficients().iter().zip([-1.0, 2.25, 2.25]) {
            close(actual, expected);
        }
        close(fit.intercept(), 2.0);
        close(
            fit.objective(),
            datafit.value(&fit.predict(&x).unwrap()).unwrap()
                + Slope::new(weights.clone())
                    .value(fit.coefficients())
                    .unwrap(),
        );
    }
}

#[test]
fn empty_feature_fits_and_weight_dimension_errors_are_explicit() {
    let x = matrix::Matrix::empty(3, 0);
    let solver = ProximalGradient::<Vec<f64>>::new();
    let fit = Problem::new(&x, Gaussian::new(&[1.0, 2.0, 3.0]), Slope::new(vec![]))
        .fit_with(&solver)
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert!(fit.coefficients().is_empty());
    close(fit.intercept(), 2.0);
    let x = matrix::Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]);
    for weights in [vec![], vec![1.0; 2], vec![-1.0]] {
        assert!(matches!(
            Problem::new(&x, Gaussian::new(&[1.0, 2.0, 3.0]), Slope::new(weights))
                .fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
}

#[test]
fn explicit_axes_apply_rank_weights_within_each_group() {
    let layout = CoefficientLayout::new(2, 2).unwrap();
    let input = [5.0, -4.0, 3.0, 2.0];
    for (axis, value, expected) in [
        (PenaltyAxis::Features, 30.0, [2.5, -2.5, 0.5, 0.5]),
        (PenaltyAxis::Responses, 32.0, [2.0, -1.0, 2.0, 1.0]),
    ] {
        let penalty = AxisPenalty::new(layout, axis, Slope::new(vec![3.0, 1.0]));
        close(penalty.value(&input).unwrap(), value);
        let mut output = [0.0; 4];
        penalty.prox(&input, 1.0, &mut output).unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
fn normalization_orders_optimization_coefficients_and_preserves_predictions() {
    let x = matrix::Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let normalized =
        matrix::Matrix::from_rows(&[&[-1.5, -0.5], &[-0.5, -2.5], &[0.5, 3.5], &[1.5, -0.5]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let datafit = Gaussian::new(&y);
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-10);
    let penalty = Slope::new(vec![0.15, 0.05]);
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
    for (&actual, expected) in lazy
        .predict(&x)
        .unwrap()
        .iter()
        .zip(explicit.predict(&normalized).unwrap())
    {
        close(actual, expected);
    }
    let beta: Vec<_> = lazy
        .coefficients()
        .iter()
        .zip([2.0, 0.5])
        .map(|(b, s)| b * s)
        .collect();
    close(
        lazy.objective(),
        datafit.value(&lazy.predict(&x).unwrap()).unwrap() + penalty.value(&beta).unwrap(),
    );
}
