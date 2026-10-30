# Binomial

`Binomial` uses the logit link and the averaged Bernoulli negative log
likelihood:

$$
  F(y,\eta)
  = \frac1n\sum_{i=1}^n \left[ \log(1 + e^{\eta_i}) - y_i\eta_i \right],
  \qquad
  p_i = \frac{1}{1 + e^{-\eta_i}}.
$$

The predictor derivatives are

$$
  \frac{\partial F}{\partial\eta_i} = \frac{p_i - y_i}{n},
  \qquad
  \frac{\partial^2 F}{\partial\eta_i^2} = \frac{p_i(1 - p_i)}{n}
                                        \leq \frac{1}{4n}.
$$

The implementation evaluates the loss and derivatives stably for large logits.
Responses must be finite and lie in $[0,1]$. Fractional values represent
proportions with equal observation weights; this API does not accept trial
counts or arbitrary observation weights. The intercept is fitted by default.

## Evaluate the loss

```rust
use shrinkage::{Binomial, Datafit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let y = [0.0, 1.0];
    let datafit = Binomial::new(&y);
    assert!((datafit.value(&[0.0, 0.0])? - 2.0_f64.ln()).abs() < 1e-14);
    Ok(())
}
```

Fitted predictions are **logits**. Apply the logistic function to obtain
probabilities. See [logistic regression](../models/logistic.md) for a stable
conversion and a complete fit.

[Proximal gradient](../reference/solvers/proximal-gradient.md) supports the
built-in proximal penalties. [IRLS](../reference/solvers/irls.md) supports L1,
ridge, and elastic net. The [hybrid solver](../reference/solvers/hybrid.md)
supports clustered lasso, chain fused lasso, and SLOPE. Unpenalized separated
data may have no finite minimizer; inspect termination and choose a regularized
model when appropriate.

API:
[`shrinkage::Binomial`](https://shrinkage.cc/packages/rust/reference/cabade7e6bb5cb30e1afe92b7012a4d0f67c7bec0235c94ef5bca1dda3c635f8).
