use faer::Mat;
use shrinkage::{ProximalPenalty, Slope};

/// Construct a known optimum using a Hadamard design and its triangular mix.
pub fn dataset(
    n: usize,
    p: usize,
    block_size: usize,
    correlation: f64,
) -> (Mat<f64>, Vec<f64>, Slope) {
    assert!(n.is_power_of_two() && p < n && p % block_size == 0);
    let hadamard = |i: usize, j: usize| {
        if (i & (j + 1)).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    };
    let x = Mat::from_fn(n, p, |i, j| {
        hadamard(i, j)
            + if j > 0 {
                correlation * hadamard(i, j - 1)
            } else {
                0.0
            }
    });
    let weights: Vec<_> = (0..p)
        .map(|j| 0.1 + 0.2 * (p - 1 - j) as f64 / (p - 1) as f64)
        .collect();
    let beta: Vec<_> = (0..p)
        .map(|j| {
            let value = 1.0 + ((p - 1 - j) / block_size * block_size) as f64 / p as f64;
            if j % 2 == 0 { value } else { -value }
        })
        .collect();
    let mut dual = vec![0.0; p];
    for start in (0..p).step_by(block_size) {
        let mean = weights[start..start + block_size].iter().sum::<f64>() / block_size as f64;
        for j in start..start + block_size {
            dual[j] = mean * beta[j].signum();
        }
    }
    // X = H B and H' H / n = I. Solving B' v = g makes the gradient
    // at beta equal -g, where g is a known sorted-L1 subgradient.
    let mut v = dual;
    for j in 1..p {
        v[j] -= correlation * v[j - 1];
    }
    let y = (0..n)
        .map(|i| {
            (0..p)
                .map(|j| x[(i, j)] * beta[j] + hadamard(i, j) * v[j])
                .sum()
        })
        .collect();
    (x, y, Slope::new(weights))
}

pub fn mapping(x: &Mat<f64>, y: &[f64], beta: &[f64], penalty: &Slope) -> f64 {
    let residual: Vec<_> = (0..y.len())
        .map(|i| (0..beta.len()).map(|j| x[(i, j)] * beta[j]).sum::<f64>() - y[i])
        .collect();
    let step = 1.0 / 4.0;
    let input: Vec<_> = (0..beta.len())
        .map(|j| {
            beta[j]
                - step
                    * (0..y.len())
                        .map(|i| x[(i, j)] * residual[i] / y.len() as f64)
                        .sum::<f64>()
        })
        .collect();
    let mut output = vec![0.0; beta.len()];
    penalty.prox(&input, step, &mut output).unwrap();
    beta.iter()
        .zip(output)
        .map(|(b, z)| (b - z).abs() / step)
        .fold(0.0, f64::max)
}
