use faer::Mat;
use shrinkage::{ClusteredLasso, ProximalPenalty};

pub fn dataset(n: usize, p: usize) -> (Mat<f64>, Vec<f64>) {
    let mut seed = 20261014_u64;
    let x = Mat::from_fn(n, p, |_, _| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    });
    let centers = [-1.0, -0.4, 0.2, 1.0];
    let y = (0..n)
        .map(|i| {
            (0..p)
                .map(|j| x[(i, j)] * centers[j % centers.len()])
                .sum::<f64>()
                + 0.01 * ((i * 7 % 11) as f64 - 5.0)
        })
        .collect();
    (x, y)
}

pub fn mapping(x: &Mat<f64>, y: &[f64], beta: &[f64], penalty: &ClusteredLasso) -> f64 {
    let curvature = x
        .col_iter()
        .flat_map(|col| col.iter().copied())
        .map(|a| a * a / y.len() as f64)
        .sum::<f64>();
    let residual: Vec<_> = (0..y.len())
        .map(|i| (0..beta.len()).map(|j| x[(i, j)] * beta[j]).sum::<f64>() - y[i])
        .collect();
    let input: Vec<_> = (0..beta.len())
        .map(|j| {
            beta[j]
                - (0..y.len())
                    .map(|i| x[(i, j)] * residual[i] / y.len() as f64)
                    .sum::<f64>()
                    / curvature
        })
        .collect();
    let mut output = vec![0.0; beta.len()];
    penalty.prox(&input, 1.0 / curvature, &mut output).unwrap();
    beta.iter()
        .zip(output)
        .map(|(b, z)| (b - z).abs() * curvature)
        .fold(0.0, f64::max)
}
