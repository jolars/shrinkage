//! Independent certificates for weighted chain fusion and its complete prox.

use shrinkage::{ChainFusedLasso, Penalty, ProximalPenalty};

fn vector(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn adjacent_edges_are_counted_once() {
    let penalty = ChainFusedLasso::new(0.3, 0.7).with_weights(vec![2.0, 0.0, 0.5]);
    let values = [2.0, -1.0, 2.0, 2.0];
    close(
        penalty.value(&values).unwrap(),
        0.3 * 7.0 + 0.7 * 6.0,
        1e-14,
    );
    assert_eq!(penalty.l1(), 0.3);
    assert_eq!(penalty.fusion(), 0.7);
    assert_eq!(penalty.weights(), Some([2.0, 0.0, 0.5].as_slice()));
    assert_eq!(ChainFusedLasso::new(0.0, 0.0).weights(), None);
    assert_eq!(ChainFusedLasso::new(0.0, 1.0).value(&values).unwrap(), 6.0);
}

#[test]
fn constructed_subgradients_certify_the_complete_prox() {
    for target in [
        vec![],
        vec![0.0_f64],
        vec![-2.0, 2.0],
        vec![-2.0, -2.0, 0.0, 0.0, 1.0],
        vec![1.0, -1.0, 1.0, 0.0, 0.0, -1.0],
        vec![1.0, 1.0, 1.0, 1.0],
    ] {
        for weights in [
            vec![1.0; target.len().saturating_sub(1)],
            (0..target.len().saturating_sub(1))
                .map(|j| [0.0, 0.125, 3.0, 0.5][j % 4])
                .collect(),
        ] {
            for (l1, fusion) in [(0.0, 0.0), (0.3, 0.0), (0.0, 0.7), (0.3, 0.7)] {
                let mut gradient: Vec<_> = target
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        l1 * if x == 0.0 {
                            (j % 3) as f64 * 0.5 - 0.5
                        } else {
                            x.signum()
                        }
                    })
                    .collect();
                for (j, &weight) in weights.iter().enumerate() {
                    let difference = target[j + 1] - target[j];
                    let dual = fusion
                        * weight
                        * if difference == 0.0 {
                            (j % 3) as f64 * 0.7 - 0.7
                        } else {
                            difference.signum()
                        };
                    gradient[j] -= dual;
                    gradient[j + 1] += dual;
                }
                let penalty = ChainFusedLasso::new(l1, fusion).with_weights(weights.clone());
                for step in [0.125, 1.0, 3.0] {
                    let input: Vec<_> = target
                        .iter()
                        .zip(&gradient)
                        .map(|(x, g)| x + step * g)
                        .collect();
                    let mut output = vec![0.0; target.len()];
                    penalty.prox(&input, step, &mut output).unwrap();
                    for (&actual, &expected) in output.iter().zip(&target) {
                        close(actual, expected, 3e-14);
                    }
                }
            }
        }
    }
}

#[test]
fn weighted_tv_duals_certify_varied_inputs() {
    for p in 1..=64 {
        for seed in 0..20 {
            let input: Vec<_> = (0..p)
                .map(|j| ((j * 37 + seed * 17) % 103) as f64 / 7.0 - 7.0)
                .collect();
            let weights: Vec<_> = (0..p - 1)
                .map(|j| ((j * 11 + seed * 5) % 23) as f64 / 3.0)
                .collect();
            let penalty = ChainFusedLasso::new(0.0, 0.7).with_weights(weights.clone());
            let mut output = vec![0.0; p];
            let step = 0.3;
            penalty.prox(&input, step, &mut output).unwrap();
            let mut dual = 0.0;
            for j in 0..p {
                dual += (output[j] - input[j]) / step;
                if j == p - 1 {
                    close(dual, 0.0, 2e-11);
                } else {
                    let bound = 0.7 * weights[j];
                    assert!(dual.abs() <= bound + 2e-11);
                    let difference = output[j + 1] - output[j];
                    if difference.abs() > 1e-11 {
                        close(dual, bound * difference.signum(), 2e-11);
                    }
                }
            }
        }
    }
}

#[test]
fn limits_empty_chains_and_reused_dimensions() {
    let penalty = ChainFusedLasso::new(0.0, 0.5);
    for (input, expected) in [
        (vec![], vec![]),
        (vec![2.0], vec![2.0]),
        (vec![0.0, -0.0], vec![0.0, 0.0]),
        (vec![1.0; 8], vec![1.0; 8]),
        (vec![-2.0, 2.0], vec![-1.5, 1.5]),
        (vec![2.0, 2.0, -2.0, -2.0], vec![1.75, 1.75, -1.75, -1.75]),
    ] {
        let mut output = vec![0.0; input.len()];
        penalty.prox(&input, 1.0, &mut output).unwrap();
        assert_eq!(output, expected);
        assert!(
            output
                .iter()
                .filter(|&&x| x == 0.0)
                .all(|x| x.is_sign_positive())
        );
    }
    let input = [-2.0, 0.25, 2.0];
    let mut output = [0.0; 3];
    ChainFusedLasso::new(0.5, 0.0)
        .prox(&input, 2.0, &mut output)
        .unwrap();
    assert_eq!(output, [-1.0, 0.0, 1.0]);
    ChainFusedLasso::new(0.0, 0.0)
        .prox(&input, 2.0, &mut output)
        .unwrap();
    assert_eq!(output, input);
    ChainFusedLasso::new(0.0, 1e10)
        .with_weights(vec![0.0, 1.0])
        .prox(&input, 1.0, &mut output)
        .unwrap();
    assert_eq!(output, [-2.0, 1.125, 1.125]);
    assert_eq!(
        ChainFusedLasso::new(0.5, 2.0)
            .with_weights(vec![])
            .value(&[])
            .unwrap(),
        0.0
    );
    assert_eq!(
        ChainFusedLasso::new(0.5, 2.0)
            .with_weights(vec![])
            .value(&[2.0])
            .unwrap(),
        1.0
    );
}

#[test]
fn validation_also_applies_at_zero_strength() {
    for (l1, fusion) in [
        (-1.0, 0.0),
        (0.0, -1.0),
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
    ] {
        let penalty = ChainFusedLasso::new(l1, fusion);
        assert!(penalty.value(&[]).is_err());
        assert!(penalty.prox(&[], 1.0, &mut []).is_err());
    }
    for weights in [
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![],
        vec![1.0, 1.0],
    ] {
        let penalty = ChainFusedLasso::new(0.0, 0.0).with_weights(weights);
        assert!(penalty.value(&[1.0, 2.0]).is_err());
        assert!(penalty.prox(&[1.0, 2.0], 1.0, &mut [0.0; 2]).is_err());
    }
    let penalty = ChainFusedLasso::new(0.0, 0.0);
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(penalty.prox(&[], step, &mut []).is_err());
    }
    assert!(penalty.value(&[f64::NAN]).is_err());
    assert!(penalty.prox(&[f64::INFINITY], 1.0, &mut [0.0]).is_err());
    assert!(penalty.prox(&[1.0], 1.0, &mut []).is_err());
}

#[test]
fn unused_terms_avoid_overflow_and_active_thresholds_reject_underflow() {
    let input = [-f64::MAX, f64::MAX];
    let penalty = ChainFusedLasso::new(0.0, f64::MAX).with_weights(vec![0.0]);
    assert_eq!(penalty.value(&input).unwrap(), 0.0);
    let mut output = [0.0; 2];
    penalty.prox(&input, 2.0, &mut output).unwrap();
    assert_eq!(output, input);
    assert!(
        ChainFusedLasso::new(0.0, f64::MAX)
            .prox(&[0.0; 2], 2.0, &mut output)
            .is_err()
    );
    assert!(
        ChainFusedLasso::new(0.0, f64::MIN_POSITIVE)
            .with_weights(vec![f64::MIN_POSITIVE])
            .prox(&[0.0; 2], 1.0, &mut output)
            .is_err()
    );
    assert!(
        ChainFusedLasso::new(f64::MIN_POSITIVE, 0.0)
            .prox(&[0.0], f64::MIN_POSITIVE, &mut [0.0])
            .is_err()
    );
}

#[test]
fn large_tube_widths_preserve_the_signal_mean() {
    let input = [1.0, 0.25, 2.0, -1.0];
    let mut output = [0.0; 4];
    ChainFusedLasso::new(0.0, 1e16)
        .prox(&input, 1.0, &mut output)
        .unwrap();
    assert_eq!(output, [0.5625; 4]);
}

#[test]
fn epigraph_qp_fixtures_independently_check_complete_proximal_maps() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("chain_fused_lasso/reference.json")).unwrap();
    for case in fixtures["proximal"].as_array().unwrap() {
        let input = vector(&case["input"]);
        let expected = vector(&case["coefficients"]);
        let weights = vector(&case["weights"]);
        let l1 = case["l1"].as_f64().unwrap();
        let fusion = case["fusion"].as_f64().unwrap();
        let step = case["step"].as_f64().unwrap();
        let penalty = ChainFusedLasso::new(l1, fusion).with_weights(weights.clone());
        let mut output = vec![0.0; input.len()];
        penalty.prox(&input, step, &mut output).unwrap();
        let objective = penalty.value(&output).unwrap()
            + output
                .iter()
                .zip(&input)
                .map(|(x, y)| (x - y).powi(2) / (2.0 * step))
                .sum::<f64>();
        close(objective, case["objective"].as_f64().unwrap(), 2e-10);
        for (&actual, &expected) in output.iter().zip(&expected) {
            close(actual, expected, 2e-6);
        }
        let l1_dual = vector(&case["l1_dual"]);
        let fusion_dual = vector(&case["fusion_dual"]);
        for j in 0..input.len() {
            assert!(l1_dual[j].abs() <= l1 + 1e-9);
            close(l1_dual[j] * expected[j], l1 * expected[j].abs(), 1e-8);
            let mut stationarity = (expected[j] - input[j]) / step + l1_dual[j];
            if j > 0 {
                stationarity += fusion_dual[j - 1];
            }
            if j < fusion_dual.len() {
                stationarity -= fusion_dual[j];
                let difference = expected[j + 1] - expected[j];
                assert!(fusion_dual[j].abs() <= fusion * weights[j] + 1e-9);
                close(
                    fusion_dual[j] * difference,
                    fusion * weights[j] * difference.abs(),
                    1e-8,
                );
            }
            close(stationarity, 0.0, 1e-8);
        }
    }
}
