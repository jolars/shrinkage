pub fn responses(latent: &[f64]) -> Vec<f64> {
    latent
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let probability = 1.0 / (1.0 + (-0.5 * (value - 1.0)).exp());
            f64::from(((i * 37) % 101) as f64 / 101.0 < probability)
        })
        .collect()
}
