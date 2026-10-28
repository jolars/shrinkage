# Logistic regression

A binomial model relates a linear predictor to a probability through the
logistic function:

$$
  \eta_i = b + \widetilde x_i^\mathsf{T}\theta,
  \qquad
  p_i = \frac{1}{1 + e^{-\eta_i}},
  \qquad
  \min_{b,\theta}\;\frac1n\sum_i[\log(1 + e^{\eta_i}) - y_i\eta_i] + P(\theta).
$$

Responses lie in $[0,1]$, with zeros and ones representing binary observations.
The intercept is unpenalized. See [binomial datafits](../datafits/binomial.md)
for fractional-response conventions.

## Fit ridge logistic regression

```rust
use faer::Mat;
use shrinkage::{Binomial, Irls, Problem, Ridge, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(6, 2, |i, j| if j == 0 { i as f64 } else { (i % 2) as f64 });
    let y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let fit = Problem::new(&x, Binomial::new(&y), Ridge::new(0.2)).fit_with(&Irls::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    let probabilities: Vec<_> = fit
        .predict(&x)?
        .iter()
        .map(|&eta| {
            if eta >= 0.0 {
                1.0 / (1.0 + (-eta).exp())
            } else {
                let e = eta.exp();
                e / (1.0 + e)
            }
        })
        .collect();
    println!("Probabilities: {probabilities:?}");
    Ok(())
}
```

`predict` returns logits. The two branches above convert them to probabilities
without overflowing the exponential for extreme predictors. Shrinkage does not
choose a classification threshold.

[IRLS](../reference/solvers/irls.md) supports L1, ridge, and elastic net.
[Proximal gradient](../reference/solvers/proximal-gradient.md) also supports the
coupled penalties, while [hybrid](../reference/solvers/hybrid.md) uses weighted
subproblems for SLOPE and the two fusion penalties. The [clustered lasso
guide](clustered-lasso.md#logistic-regression) explains hybrid logistic budgets
and diagnostics.
