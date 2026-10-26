//! Independent checks of uniform signed fusion and its complete proximal map.

use shrinkage::{ClusteredLasso, Penalty, ProximalPenalty};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn pairs(values: &[f64], l1: f64, fusion: f64) -> f64 {
    l1 * values.iter().map(|x| x.abs()).sum::<f64>()
        + fusion
            * (0..values.len())
                .flat_map(|i| (i + 1..values.len()).map(move |j| (values[i] - values[j]).abs()))
                .sum::<f64>()
}

#[test]
fn signed_pairs_are_counted_once() {
    for input in [
        vec![],
        vec![2.0],
        vec![-2.0, 2.0],
        vec![3.0, -1.0, 3.0, 0.0],
    ] {
        for (l1, fusion) in [(0.0, 0.0), (0.3, 0.0), (0.0, 0.7), (0.3, 0.7)] {
            close(
                ClusteredLasso::new(l1, fusion).value(&input).unwrap(),
                pairs(&input, l1, fusion),
                1e-14,
            );
        }
    }
}

#[test]
fn explicit_subgradients_certify_the_complete_prox() {
    // Construct inputs from an independently chosen point and a subgradient
    // using the original unordered pairs, including interior subgradients.
    for target in [
        vec![],
        vec![0.0_f64],
        vec![-2.0, 2.0],
        vec![-2.0, -2.0, 0.0, 0.0, 1.0],
        vec![1.0, -1.0, 1.0, -1.0, 0.0],
    ] {
        for (l1, fusion) in [(0.0, 0.0), (0.3, 0.0), (0.0, 0.7), (0.3, 0.7)] {
            let mut gradient: Vec<f64> = target
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    l1 * if x == 0.0 {
                        (i as f64 % 3.0 - 1.0) * 0.4
                    } else {
                        x.signum()
                    }
                })
                .collect();
            for i in 0..target.len() {
                for j in i + 1..target.len() {
                    let difference = target[i] - target[j];
                    let sign = if difference == 0.0 {
                        ((i + j) % 3) as f64 * 0.5 - 0.5
                    } else {
                        difference.signum()
                    };
                    gradient[i] += fusion * sign;
                    gradient[j] -= fusion * sign;
                }
            }
            let penalty = ClusteredLasso::new(l1, fusion);
            for step in [0.125, 1.0, 3.0] {
                let input: Vec<_> = target
                    .iter()
                    .zip(&gradient)
                    .map(|(x, g)| x + step * g)
                    .collect();
                let mut output = vec![0.0; target.len()];
                penalty.prox(&input, step, &mut output).unwrap();
                for (&actual, &expected) in output.iter().zip(&target) {
                    close(actual, expected, 2e-14);
                }
            }
        }
    }
}

#[test]
fn limits_ties_signed_zero_and_reused_dimensions() {
    let penalty = ClusteredLasso::new(0.0, 0.5);
    for (input, expected) in [
        (vec![-2.0, 2.0], vec![-1.5, 1.5]),
        (vec![0.0, -0.0], vec![0.0, 0.0]),
        (vec![1.0, 1.0, 1.0], vec![1.0; 3]),
        (vec![1.0], vec![1.0]),
        (vec![], vec![]),
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
    let mut output = [0.0; 3];
    ClusteredLasso::new(0.5, 0.0)
        .prox(&[-2.0, 0.25, 2.0], 2.0, &mut output)
        .unwrap();
    assert_eq!(output, [-1.0, 0.0, 1.0]);
    ClusteredLasso::new(0.0, 0.0)
        .prox(&[-2.0, 0.25, 2.0], 2.0, &mut output)
        .unwrap();
    assert_eq!(output, [-2.0, 0.25, 2.0]);
}

#[test]
fn invalid_inputs_are_rejected_even_in_zero_strength_limits() {
    for (l1, fusion) in [
        (-1.0, 0.0),
        (0.0, -1.0),
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
    ] {
        let penalty = ClusteredLasso::new(l1, fusion);
        assert!(penalty.value(&[]).is_err());
        assert!(penalty.prox(&[], 1.0, &mut []).is_err());
    }
    let penalty = ClusteredLasso::new(0.0, 0.0);
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(penalty.prox(&[], step, &mut []).is_err());
    }
    assert!(penalty.prox(&[1.0], 1.0, &mut []).is_err());
    assert!(penalty.value(&[f64::NAN]).is_err());
    assert!(penalty.prox(&[f64::INFINITY], 1.0, &mut [0.0]).is_err());
}

#[test]
fn unused_strengths_do_not_create_overflow() {
    assert_eq!(
        ClusteredLasso::new(0.0, f64::MAX).value(&[2.0; 3]).unwrap(),
        0.0
    );
    assert_eq!(
        ClusteredLasso::new(0.0, 0.0)
            .value(&[-f64::MAX, f64::MAX])
            .unwrap(),
        0.0
    );
    let mut output = [0.0; 2];
    ClusteredLasso::new(0.0, 0.0)
        .prox(&[-f64::MAX, f64::MAX], 2.0, &mut output)
        .unwrap();
    assert_eq!(output, [-f64::MAX, f64::MAX]);
    assert!(
        ClusteredLasso::new(0.0, f64::MAX)
            .prox(&[0.0; 3], 2.0, &mut [0.0; 3])
            .is_err()
    );
}

#[test]
fn exhaustive_ordered_partitions_independently_minimize_the_proximal_objective() {
    for p in 1..=6 {
        for seed in 0..12 {
            let input: Vec<_> = (0..p)
                .map(|j| ((j * 17 + seed * 11) % 23) as f64 / 3.0 - 4.0)
                .collect();
            let mut sorted = input.clone();
            sorted.sort_by(f64::total_cmp);
            for (l1, fusion, step) in [(0.2, 0.3, 0.7), (0.0, 0.5, 2.0), (0.4, 0.0, 0.25)] {
                let objective = |candidate: &[f64]| {
                    pairs(candidate, l1, fusion)
                        + candidate
                            .iter()
                            .zip(&sorted)
                            .map(|(z, v)| (z - v).powi(2) / (2.0 * step))
                            .sum::<f64>()
                };
                let mut best = f64::INFINITY;
                for cuts in 0..1 << (p - 1) {
                    let mut candidate = vec![0.0; p];
                    let mut start = 0;
                    for end in 1..=p {
                        if end != p && cuts & (1 << (end - 1)) == 0 {
                            continue;
                        }
                        let mean = sorted[start..end].iter().sum::<f64>() / (end - start) as f64;
                        let linear = fusion * (start as f64 + end as f64 - p as f64);
                        let center = mean - step * linear;
                        let level = (center.abs() - step * l1).max(0.0).copysign(center);
                        candidate[start..end].fill(level);
                        start = end;
                    }
                    if candidate.windows(2).all(|pair| pair[0] <= pair[1]) {
                        best = best.min(objective(&candidate));
                    }
                }
                let penalty = ClusteredLasso::new(l1, fusion);
                let mut actual = vec![0.0; p];
                penalty.prox(&input, step, &mut actual).unwrap();
                actual.sort_by(f64::total_cmp);
                close(objective(&actual), best, 2e-13);
            }
        }
    }
}
