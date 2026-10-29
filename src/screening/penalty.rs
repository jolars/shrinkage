//! Complete dual polytopes and conservative fixed-zero certificates.

use super::GapSafePenalty;
use crate::error::invalid;
use crate::{ChainFusedLasso, ClusteredLasso, FitError, Penalty, Slope};

// Directed adjacent floats keep arithmetic bounds conservative on the MSRV,
// which predates the standard library's next-up and next-down methods.
fn up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(if value > 0.0 {
            value.to_bits() + 1
        } else {
            value.to_bits() - 1
        })
    }
}

fn down(value: f64) -> f64 {
    -up(-value)
}

fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        up(a + b)
    }
}

fn add_down(a: f64, b: f64) -> f64 {
    -add_up(-a, -b)
}

fn mul_down(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        down(a * b).max(0.0)
    }
}

fn ratio_up(numerator: f64, denominator: f64) -> f64 {
    if numerator <= 0.0 {
        0.0
    } else if denominator == 0.0 {
        f64::INFINITY
    } else {
        up(numerator / denominator)
    }
}

fn validate_bounds(correlations: &[f64], errors: &[f64]) -> Result<(), FitError> {
    if correlations.len() != errors.len() {
        return Err(invalid("dual correlation and error dimensions must match"));
    }
    if correlations.iter().any(|g| !g.is_finite()) || errors.iter().any(|e| e.is_nan() || *e < 0.0)
    {
        return Err(invalid(
            "dual correlations must be finite and errors nonnegative",
        ));
    }
    Ok(())
}

fn validate_mask(correlations: &[f64], errors: &[f64], zeroed: &[bool]) -> Result<(), FitError> {
    validate_bounds(correlations, errors)?;
    if correlations.len() != zeroed.len() {
        return Err(invalid("zero mask length must match the coefficients"));
    }
    Ok(())
}

fn clustered_capacity(penalty: &ClusteredLasso, length: usize, count: usize) -> f64 {
    add_down(
        mul_down(penalty.l1(), count as f64),
        mul_down(
            mul_down(penalty.fusion(), count as f64),
            (length - count) as f64,
        ),
    )
}

impl GapSafePenalty for ClusteredLasso {
    fn has_sparsity(&self) -> bool {
        self.l1() > 0.0
    }

    fn dual_scaling(&self, correlations: &[f64], errors: &[f64]) -> Result<Option<f64>, FitError> {
        self.validate_input(correlations)?;
        validate_bounds(correlations, errors)?;
        let mut scale = 1.0_f64;
        for sign in [-1.0, 1.0] {
            let bound = |j: usize| add_up(sign * correlations[j], errors[j]);
            self.sorted_bound_indices(correlations.len(), bound, |order| {
                let mut sum = 0.0;
                for (rank, &j) in order.iter().enumerate() {
                    sum = add_up(sum, bound(j));
                    scale = scale.max(ratio_up(
                        sum,
                        clustered_capacity(self, order.len(), rank + 1),
                    ));
                }
            });
        }
        Ok(Some(scale))
    }

    fn dual_feasible(&self, correlations: &[f64], errors: &[f64]) -> Result<bool, FitError> {
        Ok(self
            .dual_scaling(correlations, errors)?
            .is_some_and(|scale| scale <= 1.0))
    }

    fn screen_zeros(
        &self,
        correlations: &[f64],
        errors: &[f64],
        zeroed: &mut [bool],
    ) -> Result<(), FitError> {
        self.validate_input(correlations)?;
        validate_mask(correlations, errors, zeroed)?;
        if !self.has_sparsity() {
            return Ok(());
        }
        let fusion = if correlations.len() <= 1 || self.fusion() == 0.0 {
            0.0
        } else {
            up(self.fusion() * (correlations.len() - 1) as f64)
        };
        let threshold = add_down(self.l1(), -fusion);
        // Every incident fusion subgradient has magnitude at most its strength.
        // This bound makes the L1 subgradient interior for any feasible split.
        for ((&g, &error), zero) in correlations.iter().zip(errors).zip(zeroed) {
            *zero |= add_up(g.abs(), error) < threshold;
        }
        Ok(())
    }

    fn prox_zeroed(
        &self,
        input: &[f64],
        zeroed: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        self.prox_screened(input, zeroed, step, output)
    }
}

fn chain_capacity(penalty: &ChainFusedLasso, edge: usize, length: usize) -> f64 {
    if edge == 0 || edge == length {
        0.0
    } else {
        mul_down(penalty.fusion(), penalty.weight(edge - 1))
    }
}

impl GapSafePenalty for ChainFusedLasso {
    fn has_sparsity(&self) -> bool {
        self.l1() > 0.0
    }

    fn dual_feasible(&self, correlations: &[f64], errors: &[f64]) -> Result<bool, FitError> {
        self.validate_input(correlations)?;
        validate_bounds(correlations, errors)?;
        if errors.iter().any(|e| !e.is_finite()) {
            return Ok(false);
        }
        let (mut lower, mut upper) = (0.0, 0.0);
        for (j, (&g, &error)) in correlations.iter().zip(errors).enumerate() {
            let capacity = chain_capacity(self, j + 1, correlations.len());
            lower = (-capacity).max(add_up(add_up(add_up(lower, -self.l1()), -g), error));
            upper = capacity.min(add_down(add_down(add_down(upper, self.l1()), -g), -error));
            if lower > upper {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn screen_zeros(
        &self,
        correlations: &[f64],
        errors: &[f64],
        zeroed: &mut [bool],
    ) -> Result<(), FitError> {
        self.validate_input(correlations)?;
        validate_mask(correlations, errors, zeroed)?;
        if !self.has_sparsity() || errors.iter().any(|e| !e.is_finite()) {
            return Ok(());
        }
        let length = correlations.len();
        self.with_screening_bounds(length + 1, |forward_lower, forward_upper| {
            forward_lower[0] = 0.0;
            forward_upper[0] = 0.0;
            for j in 0..length {
                let capacity = chain_capacity(self, j + 1, length);
                forward_lower[j + 1] = (-capacity).max(add_up(
                    add_up(add_up(forward_lower[j], -self.l1()), -correlations[j]),
                    errors[j],
                ));
                forward_upper[j + 1] = capacity.min(add_down(
                    add_down(add_down(forward_upper[j], self.l1()), -correlations[j]),
                    -errors[j],
                ));
            }
            let (mut backward_lower, mut backward_upper) = (0.0, 0.0);
            for j in (0..length).rev() {
                let upper = add_up(
                    add_up(add_up(correlations[j], errors[j]), -forward_upper[j]),
                    backward_lower,
                );
                let lower = add_down(
                    add_down(add_down(correlations[j], -errors[j]), -forward_lower[j]),
                    backward_upper,
                );
                // The left and right reachable intervals certify an interior L1
                // component at every optimal correlation in the safe region.
                zeroed[j] |= upper < self.l1() && lower > -self.l1();
                let capacity = chain_capacity(self, j, length);
                backward_lower = (-capacity).max(add_up(
                    add_up(add_up(backward_lower, -self.l1()), correlations[j]),
                    errors[j],
                ));
                backward_upper = capacity.min(add_down(
                    add_down(add_down(backward_upper, self.l1()), correlations[j]),
                    -errors[j],
                ));
            }
        });
        Ok(())
    }

    fn prox_zeroed(
        &self,
        input: &[f64],
        zeroed: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        self.prox_screened(input, zeroed, step, output)
    }
}

impl GapSafePenalty for Slope {
    fn has_sparsity(&self) -> bool {
        self.weights().first().is_some_and(|weight| *weight > 0.0)
    }

    fn dual_scaling(&self, correlations: &[f64], errors: &[f64]) -> Result<Option<f64>, FitError> {
        self.validate_input(correlations)?;
        validate_bounds(correlations, errors)?;
        let bound = |j: usize| add_up(correlations[j].abs(), errors[j]);
        let scale = self.sorted_bound_indices(correlations.len(), bound, |order| {
            let (mut sum, mut capacity, mut scale) = (0.0, 0.0, 1.0_f64);
            for (&j, &weight) in order.iter().zip(self.weights()) {
                sum = add_up(sum, bound(j));
                capacity = add_down(capacity, weight).max(0.0);
                scale = scale.max(ratio_up(sum, capacity));
            }
            scale
        });
        Ok(Some(scale))
    }

    fn dual_feasible(&self, correlations: &[f64], errors: &[f64]) -> Result<bool, FitError> {
        Ok(self
            .dual_scaling(correlations, errors)?
            .is_some_and(|scale| scale <= 1.0))
    }

    fn screen_zeros(
        &self,
        correlations: &[f64],
        errors: &[f64],
        zeroed: &mut [bool],
    ) -> Result<(), FitError> {
        self.validate_input(correlations)?;
        validate_mask(correlations, errors, zeroed)?;
        if !self.has_sparsity() {
            return Ok(());
        }
        let bound = |j: usize| add_up(correlations[j].abs(), errors[j]);
        self.with_screening_order(zeroed, bound, |order, thresholds, zeroed| {
            let (mut excess, mut threshold) = (0.0, f64::INFINITY);
            for ((&j, &weight), saved) in order.iter().zip(self.weights()).zip(&mut *thresholds) {
                threshold = threshold.min(add_down(weight, excess));
                *saved = threshold;
                excess = add_down(excess, add_down(weight, -bound(j))).max(0.0);
            }
            // This running maximum evaluates every suffix bound for each prefix.
            // After a safe tail removal, the remaining objective uses leading weights.
            for (&j, &threshold) in order.iter().zip(thresholds.iter()).rev() {
                if bound(j) >= threshold {
                    break;
                }
                zeroed[j] = true;
            }
        });
        Ok(())
    }

    fn prox_zeroed(
        &self,
        input: &[f64],
        zeroed: &[bool],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        self.prox_screened(input, zeroed, step, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProximalPenalty;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 2e-12 * (1.0 + expected.abs()),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn strict_boundaries_and_zero_strengths_preserve_existing_masks() {
        let penalties: Vec<Box<dyn GapSafePenalty>> = vec![
            Box::new(ClusteredLasso::new(1.0, 0.0)),
            Box::new(ChainFusedLasso::new(1.0, 0.0)),
            Box::new(Slope::new(vec![1.0; 3])),
        ];
        for penalty in penalties {
            let mut mask = [false, false, true];
            penalty
                .screen_zeros(&[1.0, 0.5, 0.2], &[0.0, 0.1, 0.0], &mut mask)
                .unwrap();
            assert!(!mask[0]);
            assert!(mask[1]);
            assert!(mask[2]);
            let mut boundary = [false; 3];
            penalty
                .screen_zeros(&[0.5; 3], &[0.5; 3], &mut boundary)
                .unwrap();
            assert_eq!(boundary, [false; 3]);
        }
        let penalties: Vec<Box<dyn GapSafePenalty>> = vec![
            Box::new(ClusteredLasso::new(0.0, 3.0)),
            Box::new(ChainFusedLasso::new(0.0, 3.0)),
            Box::new(Slope::new(vec![0.0; 3])),
        ];
        for penalty in penalties {
            assert!(!penalty.has_sparsity());
            assert!(penalty.dual_feasible(&[0.0; 3], &[0.0; 3]).unwrap());
            let mut mask = [false, true, false];
            penalty
                .screen_zeros(&[0.0; 3], &[0.0; 3], &mut mask)
                .unwrap();
            assert_eq!(mask, [false, true, false]);
        }
    }

    #[test]
    fn clustered_dual_scaling_matches_all_signed_subsets() {
        let penalty = ClusteredLasso::new(0.7, 0.4);
        for seed in 0..100 {
            let g: Vec<_> = (0..5)
                .map(|j| ((seed * 13 + j * 7) % 43) as f64 / 5.0 - 4.0)
                .collect();
            let e = [0.01, 0.03, 0.0, 0.1, 0.04];
            let mut expected = 1.0_f64;
            for subset in 1_usize..32 {
                let count = subset.count_ones() as usize;
                let capacity = 0.7 * count as f64 + 0.4 * (count * (5 - count)) as f64;
                for sign in [-1.0, 1.0] {
                    let value: f64 = (0..5)
                        .filter(|j| subset & (1 << j) != 0)
                        .map(|j| sign * g[j] + e[j])
                        .sum();
                    expected = expected.max(value / capacity);
                }
            }
            let actual = penalty.dual_scaling(&g, &e).unwrap().unwrap();
            assert!(actual >= expected - 1e-14);
            close(actual, expected);
            let scaled: Vec<_> = g.iter().map(|g| g / (actual * (1.0 + 1e-12))).collect();
            let errors: Vec<_> = e.iter().map(|e| e / (actual * (1.0 + 1e-12))).collect();
            assert!(penalty.dual_feasible(&scaled, &errors).unwrap());
        }
    }

    #[test]
    fn slope_polar_uses_complete_prefixes_with_equal_and_zero_weights() {
        for weights in [
            vec![2.0, 2.0, 2.0],
            vec![3.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ] {
            let penalty = Slope::new(weights.clone());
            let g: [f64; 3] = [-2.0, 4.0, 1.0];
            let errors = [0.02, 0.01, 0.03];
            let mut bounds: Vec<_> = g.iter().zip(errors).map(|(g, e)| g.abs() + e).collect();
            bounds.sort_unstable_by(|a, b| b.total_cmp(a));
            let (mut expected, mut numerator, mut denominator) = (1.0_f64, 0.0, 0.0);
            for (&bound, &weight) in bounds.iter().zip(&weights) {
                numerator += bound;
                denominator += weight;
                expected = expected.max(numerator / denominator);
            }
            close(
                penalty.dual_scaling(&g, &errors).unwrap().unwrap(),
                expected,
            );
        }
    }

    #[test]
    fn slope_recurrence_matches_exhaustive_prefix_suffix_tests() {
        for length in 1..=7 {
            for seed in 0..80 {
                let mut weights: Vec<_> = (0..length)
                    .map(|j| ((seed * 17 + j * 11) % 19) as f64 / 5.0)
                    .collect();
                weights.sort_unstable_by(|a, b| b.total_cmp(a));
                let g: Vec<_> = (0..length)
                    .map(|j| ((seed * 7 + j * 3) % 29) as f64 / 7.0 - 1.5)
                    .collect();
                let errors: Vec<_> = (0..length)
                    .map(|j| ((seed + j) % 3) as f64 * 0.01)
                    .collect();
                let bounds: Vec<_> = g.iter().zip(&errors).map(|(g, e)| g.abs() + e).collect();
                let mut expected = vec![false; length];
                for j in 0..length {
                    let mut others: Vec<_> = bounds
                        .iter()
                        .enumerate()
                        .filter(|&(k, _)| k != j)
                        .map(|(_, &v)| v)
                        .collect();
                    others.sort_unstable_by(|a, b| b.total_cmp(a));
                    expected[j] = (1..=length).all(|q| {
                        (1..=q).any(|s| {
                            bounds[j] + others[s - 1..q - 1].iter().sum::<f64>()
                                < weights[s - 1..q].iter().sum::<f64>() - 1e-12
                        })
                    });
                }
                let mut actual = vec![false; length];
                Slope::new(weights)
                    .screen_zeros(&g, &errors, &mut actual)
                    .unwrap();
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!(!actual || expected, "length={length}, seed={seed}");
                }
            }
        }
        let mut mask = [false; 3];
        Slope::new(vec![2.0, 1.0, 0.0])
            .screen_zeros(&[2.0, 0.7, 0.1], &[0.0; 3], &mut mask)
            .unwrap();
        assert_eq!(mask, [false, true, true]);
    }

    #[test]
    fn chain_zero_test_certifies_the_whole_region() {
        let penalty = ChainFusedLasso::new(1.0, 3.0);
        let center = [3.0, -3.0];
        assert!(penalty.dual_feasible(&center, &[0.0; 2]).unwrap());
        let mut narrow = [false; 2];
        penalty
            .screen_zeros(&center, &[0.9; 2], &mut narrow)
            .unwrap();
        assert_eq!(narrow, [true; 2]);
        let mut wide = [false; 2];
        penalty.screen_zeros(&center, &[1.1; 2], &mut wide).unwrap();
        assert_eq!(wide, [false; 2]);
        // The wider region includes the dual supporting a nonzero contrast.
        let mut output = [0.0; 2];
        penalty.prox(&[5.0, -5.0], 1.0, &mut output).unwrap();
        assert_eq!(output, [1.0, -1.0]);
    }

    #[test]
    fn chain_membership_matches_every_contiguous_cut() {
        let penalty = ChainFusedLasso::new(0.6, 1.0).with_weights(vec![0.8, 0.0, 1.3]);
        let capacities = [0.0, 0.8, 0.0, 1.3, 0.0];
        for seed in 0..300 {
            let g: Vec<_> = (0..4)
                .map(|j| ((seed * 13 + j * 17) % 41) as f64 / 13.0 - 1.5)
                .collect();
            let errors = [0.01, 0.02, 0.015, 0.005];
            let expected = (0..4).all(|start| {
                (start + 1..=4).all(|end| {
                    let sum: f64 = g[start..end].iter().sum();
                    let error: f64 = errors[start..end].iter().sum();
                    sum.abs() + error
                        < 0.6 * (end - start) as f64 + capacities[start] + capacities[end] - 1e-12
                })
            });
            assert_eq!(
                penalty.dual_feasible(&g, &errors).unwrap(),
                expected,
                "seed={seed}"
            );
        }
    }

    #[test]
    fn zeroed_proxes_preserve_interactions_and_original_rank_weights() {
        let mut output = [f64::NAN; 4];
        ClusteredLasso::new(0.2, 0.3)
            .prox_zeroed(
                &[-2.0, 99.0, 3.0, 0.5],
                &[false, true, false, false],
                1.0,
                &mut output,
            )
            .unwrap();
        for (&actual, expected) in output.iter().zip([-0.9, 0.0, 1.9, 0.0]) {
            close(actual, expected);
        }
        ChainFusedLasso::new(0.2, 0.7)
            .with_weights(vec![1.0, 2.0, 0.5])
            .prox_zeroed(
                &[3.0, 99.0, -3.0, 4.0],
                &[false, true, false, false],
                1.0,
                &mut output,
            )
            .unwrap();
        for (&actual, expected) in output.iter().zip([2.1, 0.0, -1.05, 3.45]) {
            close(actual, expected);
        }
        let slope = Slope::new(vec![3.0, 2.0, 1.0, 0.0]);
        slope
            .prox_zeroed(
                &[4.0, 99.0, -4.0, 0.5],
                &[false, true, false, false],
                1.0,
                &mut output,
            )
            .unwrap();
        for (&actual, expected) in output.iter().zip([1.5, 0.0, -1.5, 0.0]) {
            close(actual, expected);
        }
        let mut fresh = [0.0; 4];
        slope
            .prox(&[3.0, 2.0, -1.0, 0.5], 0.4, &mut output)
            .unwrap();
        Slope::new(vec![3.0, 2.0, 1.0, 0.0])
            .prox(&[3.0, 2.0, -1.0, 0.5], 0.4, &mut fresh)
            .unwrap();
        assert_eq!(output, fresh);
    }

    #[test]
    fn constructed_original_subgradients_certify_restricted_fusion_proxes() {
        for bits in 0..32 {
            let mask: Vec<_> = (0..5).map(|j| bits & (1 << j) != 0).collect();
            let target: Vec<_> = [-2.0_f64, -2.0, 0.0, 1.0, 3.0]
                .iter()
                .zip(&mask)
                .map(|(&value, &zero)| if zero { 0.0 } else { value })
                .collect();
            for chain in [false, true] {
                let (l1, fusion) = (0.3, 0.7);
                let weights = [1.0, 0.0, 2.0, 0.5];
                let mut subgradient: Vec<_> = target
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        l1 * if v == 0.0 {
                            j as f64 * 0.2 - 0.4
                        } else {
                            v.signum()
                        }
                    })
                    .collect();
                for i in 0..5 {
                    for j in i + 1..5 {
                        if chain && j != i + 1 {
                            continue;
                        }
                        let difference = target[j] - target[i];
                        let sign = if difference == 0.0 {
                            ((i + j) % 3) as f64 * 0.5 - 0.5
                        } else {
                            difference.signum()
                        };
                        let dual = fusion * if chain { weights[i] } else { 1.0 } * sign;
                        subgradient[i] -= dual;
                        subgradient[j] += dual;
                    }
                }
                let penalty: Box<dyn GapSafePenalty> = if chain {
                    Box::new(ChainFusedLasso::new(l1, fusion).with_weights(weights.to_vec()))
                } else {
                    Box::new(ClusteredLasso::new(l1, fusion))
                };
                for step in [0.125, 1.0, 3.0] {
                    let input: Vec<_> = target
                        .iter()
                        .zip(&subgradient)
                        .zip(&mask)
                        .map(|((&v, &g), &zero)| if zero { 100.0 } else { v + step * g })
                        .collect();
                    let mut output = vec![0.0; 5];
                    penalty
                        .prox_zeroed(&input, &mask, step, &mut output)
                        .unwrap();
                    for (&actual, &expected) in output.iter().zip(&target) {
                        close(actual, expected);
                    }
                    let mut certified = mask.clone();
                    penalty
                        .screen_zeros(&subgradient, &[0.0; 5], &mut certified)
                        .unwrap();
                    for (&value, &zero) in target.iter().zip(&certified) {
                        assert!(!zero || value == 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn slope_restrictions_and_zero_tests_preserve_constructed_nonzero_optima() {
        let weights = [3.0, 2.0, 1.0, 0.0, 0.0];
        let penalty = Slope::new(weights.to_vec());
        for bits in 0..32 {
            let mask: Vec<_> = (0..5).map(|j| bits & (1 << j) != 0).collect();
            let target: Vec<_> = [-2.0_f64, 2.0, 0.0, 1.0, -1.0]
                .iter()
                .zip(&mask)
                .map(|(&value, &zero)| if zero { 0.0 } else { value })
                .collect();
            let mut order: Vec<_> = (0..5).collect();
            order.sort_unstable_by(|&a, &b| target[b].abs().total_cmp(&target[a].abs()));
            let mut subgradient = [0.0; 5];
            let mut start = 0;
            while start < 5 {
                let mut end = start + 1;
                while end < 5 && target[order[end]].abs() == target[order[start]].abs() {
                    end += 1;
                }
                let weight = weights[start..end].iter().sum::<f64>() / (end - start) as f64;
                for &j in &order[start..end] {
                    subgradient[j] = if target[j] == 0.0 {
                        0.0
                    } else {
                        target[j].signum() * weight
                    };
                }
                start = end;
            }
            let mut certified = mask.clone();
            penalty
                .screen_zeros(&subgradient, &[0.0; 5], &mut certified)
                .unwrap();
            for (&value, &zero) in target.iter().zip(&certified) {
                assert!(!zero || value == 0.0);
            }
            for step in [0.125, 1.0, 3.0] {
                let input: Vec<_> = target
                    .iter()
                    .zip(&subgradient)
                    .zip(&mask)
                    .map(|((&v, &g), &zero)| if zero { 100.0 } else { v + step * g })
                    .collect();
                let mut output = [0.0; 5];
                penalty
                    .prox_zeroed(&input, &mask, step, &mut output)
                    .unwrap();
                for (&actual, &expected) in output.iter().zip(&target) {
                    close(actual, expected);
                }
            }
        }
    }

    #[test]
    fn restrictions_validate_all_coordinates_and_degenerate_masks() {
        let penalties: Vec<Box<dyn GapSafePenalty>> = vec![
            Box::new(ClusteredLasso::new(0.3, 0.7)),
            Box::new(ChainFusedLasso::new(0.3, 0.7)),
            Box::new(Slope::new(vec![3.0, 1.0])),
        ];
        for penalty in penalties {
            assert!(
                penalty
                    .prox_zeroed(&[f64::NAN, 1.0], &[true, false], 1.0, &mut [0.0; 2])
                    .is_err()
            );
            assert!(
                penalty
                    .prox_zeroed(&[0.0; 2], &[true], 1.0, &mut [0.0; 2])
                    .is_err()
            );
            assert!(
                penalty
                    .prox_zeroed(&[0.0; 2], &[true; 2], 0.0, &mut [0.0; 2])
                    .is_err()
            );
            let mut output = [f64::NAN; 2];
            penalty
                .prox_zeroed(&[10.0, -10.0], &[true; 2], 1.0, &mut output)
                .unwrap();
            assert_eq!(output, [0.0; 2]);
            assert!(output.iter().all(|value| value.is_sign_positive()));
        }
    }
}
