//! Exact dyadic balancing of observation-scale intercept certificates.

use crate::FitError;
use crate::error::invalid;

fn power_of_two(exponent: i32) -> f64 {
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1_u64 << (exponent + 1074))
    }
}

fn exponent(value: f64) -> i32 {
    let bits = value.to_bits();
    let encoded = ((bits >> 52) & 0x7ff) as i32;
    if encoded == 0 {
        63 - bits.leading_zeros() as i32 - 1074
    } else {
        encoded - 1023
    }
}

fn bounds(response: Option<f64>, n: f64, grid: f64, limit: i64) -> (i64, i64) {
    let Some(y) = response else {
        return (-limit, limit);
    };
    let lower = (y - 1.0) / n;
    let upper = y / n;
    let mut first = (lower / grid).ceil().max(-(limit as f64)) as i64;
    let mut last = (upper / grid).floor().min(limit as f64) as i64;
    // Division can round across an endpoint. Verify the stored dyadic value.
    if first as f64 * grid < lower {
        first += 1;
    }
    if last as f64 * grid > upper {
        last -= 1;
    }
    (first, last)
}

/// Put the stored certificate exactly on the intercept hyperplane.
///
/// All entries share a dyadic grid, and their integer absolute sum is at most
/// `2^50`. Consequently every floating-point partial sum is exact. Binomial
/// entries also remain inside their observation-specific conjugate domains.
pub(crate) fn balance(dual: &mut [f64], response: Option<&[f64]>) -> Result<(), FitError> {
    if response.is_some_and(|values| values.len() != dual.len()) {
        return Err(invalid(
            "intercept certificate response dimensions must match",
        ));
    }
    if dual.iter().any(|value| !value.is_finite())
        || response.is_some_and(|values| {
            values
                .iter()
                .any(|y| !y.is_finite() || !(0.0..=1.0).contains(y))
        })
    {
        return Err(invalid(
            "intercept certificates require finite values and valid responses",
        ));
    }
    let mut maximum = dual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if dual.is_empty() || maximum == 0.0 {
        return Ok(());
    }
    let count_bits = usize::BITS - (dual.len() - 1).leading_zeros();
    if count_bits > 50 {
        dual.fill(0.0);
        return Ok(());
    }
    let n = dual.len() as f64;
    // Keep even a sum of absolute values representable before quantization.
    let maximum_safe = (f64::MAX / n) * 0.25;
    if maximum > maximum_safe {
        let factor = maximum_safe / maximum;
        for value in dual.iter_mut() {
            *value *= factor;
        }
        maximum = dual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    }
    let magnitude_bits = 50 - count_bits;
    let grid_exponent = (exponent(maximum) - magnitude_bits as i32 + 1).max(-1074);
    let grid = power_of_two(grid_exponent);
    let limit = 1_i64 << magnitude_bits;
    let mut sum = 0_i64;
    for (i, value) in dual.iter_mut().enumerate() {
        let (lower, upper) = bounds(response.map(|values| values[i]), n, grid, limit);
        if lower > upper {
            dual.fill(0.0);
            return Ok(());
        }
        let integer = ((*value / grid).round() as i64).clamp(lower, upper);
        *value = integer as f64 * grid;
        sum += integer;
    }
    let mut correction = -sum;
    for (i, value) in dual.iter_mut().enumerate() {
        let (lower, upper) = bounds(response.map(|values| values[i]), n, grid, limit);
        let integer = (*value / grid) as i64;
        let change = correction.clamp(lower - integer, upper - integer);
        *value = (integer + change) as f64 * grid;
        correction -= change;
        if correction == 0 {
            break;
        }
    }
    if correction != 0 || dual.iter().sum::<f64>() != 0.0 {
        dual.fill(0.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_sum(values: &[f64]) {
        let unit = values
            .iter()
            .filter(|&&value| value != 0.0)
            .map(|value| {
                let bits = value.abs().to_bits();
                let encoded = ((bits >> 52) & 0x7ff) as i32;
                let mantissa = if encoded == 0 {
                    bits
                } else {
                    (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52)
                };
                let base = if encoded == 0 {
                    -1074
                } else {
                    encoded - 1023 - 52
                };
                base + mantissa.trailing_zeros() as i32
            })
            .min();
        if let Some(unit) = unit {
            let unit = power_of_two(unit);
            let integer_sum: i128 = values.iter().map(|&value| (value / unit) as i128).sum();
            assert_eq!(integer_sum, 0);
        }
        assert_eq!(values.iter().sum::<f64>(), 0.0);
        assert_eq!(values.iter().rev().sum::<f64>(), 0.0);
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(sorted.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn gaussian_certificates_balance_large_and_subnormal_values_exactly() {
        for mut values in [
            vec![1e200, -1e200, 3e184, -2e184],
            vec![f64::MAX / 3.0, f64::MAX / 3.0, -f64::MAX / 3.0, -1.0],
            vec![f64::from_bits(1), -f64::from_bits(1), f64::from_bits(1)],
            vec![0.25, -0.125, -0.12499999999999999],
        ] {
            balance(&mut values, None).unwrap();
            assert!(values.iter().any(|&value| value != 0.0));
            exact_sum(&values);
        }
    }

    #[test]
    fn binomial_certificates_preserve_binary_and_fractional_domains() {
        for (response, mut dual) in [
            (vec![0.0, 1.0], vec![-5e-301, 5e-301]),
            (
                vec![0.0, 1.0, 1e-12, 0.25, 0.5, 1.0],
                vec![-0.1, 0.1, -0.16, 0.04, -0.08, 0.0],
            ),
            (vec![0.1, 0.2, 0.9], vec![-0.3, 0.2 / 3.0, 0.9 / 3.0]),
        ] {
            balance(&mut dual, Some(&response)).unwrap();
            exact_sum(&dual);
            for (&u, &y) in dual.iter().zip(&response) {
                assert!(u >= (y - 1.0) / dual.len() as f64);
                assert!(u <= y / dual.len() as f64);
            }
        }
    }

    #[test]
    fn invalid_values_are_rejected_and_empty_certificates_are_allowed() {
        assert!(balance(&mut [f64::NAN], None).is_err());
        assert!(balance(&mut [0.0], Some(&[])).is_err());
        assert!(balance(&mut [0.0], Some(&[2.0])).is_err());
        balance(&mut [], None).unwrap();
    }
}
