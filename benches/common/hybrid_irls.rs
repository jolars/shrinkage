use faer::Mat;
use shrinkage::{Binomial, Datafit, HybridFit, ProximalPenalty};

pub fn dataset(n: usize, p: usize) -> (Mat<f64>, Vec<f64>) {
    let mut seed = 20261026_u64;
    let x = Mat::from_fn(n, p, |_, _| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    });
    let y = (0..n)
        .map(|i| {
            let eta = 0.3
                + (0..p)
                    .map(|j| x[(i, j)] * [-0.4, -0.4, 0.2, 0.2][j % 4])
                    .sum::<f64>();
            1.0 / (1.0 + (-eta).exp())
        })
        .collect();
    (x, y)
}

pub fn check<P: ProximalPenalty>(x: &Mat<f64>, y: &[f64], fit: &HybridFit, penalty: &P) {
    let eta = fit.predict(x).unwrap();
    let g: Vec<_> = eta
        .iter()
        .zip(y)
        .map(|(&eta, &y)| {
            let tail = (-eta.abs()).exp();
            (if eta >= 0.0 {
                (1.0 - y) - tail / (1.0 + tail)
            } else {
                tail / (1.0 + tail) - y
            }) / x.nrows() as f64
        })
        .collect();
    let input: Vec<_> = fit
        .coefficients()
        .iter()
        .enumerate()
        .map(|(j, &b)| b - fit.step_size() * (0..x.nrows()).map(|i| x[(i, j)] * g[i]).sum::<f64>())
        .collect();
    let mut prox = vec![0.0; x.ncols()];
    penalty.prox(&input, fit.step_size(), &mut prox).unwrap();
    let mapping = fit
        .coefficients()
        .iter()
        .zip(prox)
        .map(|(b, z)| (b - z).abs() / fit.step_size())
        .fold(g.iter().sum::<f64>().abs(), f64::max);
    assert!((fit.stopping_value() - mapping).abs() < 1e-12);
    assert!(
        (fit.objective()
            - Binomial::new(y).value(&eta).unwrap()
            - penalty.value(fit.coefficients()).unwrap())
        .abs()
            < 1e-12
    );
}
