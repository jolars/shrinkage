# Elastic net

Elastic net combines sparse selection with quadratic shrinkage. Its rounded
diamond retains corners on the coordinate axes. Supply the two strengths
independently with `ElasticNet::new(l1, l2)`. Setting either strength to zero
recovers ridge or lasso, respectively.

$$
  P(\theta) = \lambda_1\|\theta\|_1 + \frac{\lambda_2}{2}\|\theta\|_2^2.
$$

## Constraint region

![Feasible region for elastic net; the shaded set includes its
boundary.](figures/elastic-net.svg)

$\lambda_1 = 0.5$ and $\lambda_2 = 1$:
$0.5\|\theta\|_1 + 0.5\|\theta\|_2^2 \leq 1$. See the [plot
conventions](index.md#read-the-constraint-plots) for the fixed budget and
coefficient scale.

## Use the penalty

```rust
use shrinkage::{ElasticNet, Penalty, ProximalPenalty};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let penalty = ElasticNet::from_alpha(0.5, 0.7);
    let coefficients = [2.0, -1.0];
    println!("Penalty: {}", penalty.value(&coefficients)?);
    let mut shrunk = [0.0; 2];
    penalty.prox(&coefficients, 1.0, &mut shrunk)?;
    println!("Proximal result: {shrunk:?}");
    Ok(())
}
```

Alternatively, `ElasticNet::from_alpha(lambda, alpha)` uses
$\lambda_1 = \lambda\alpha$ and $\lambda_2 = \lambda(1 - \alpha)$, where
$0 \leq \alpha \leq 1$. The proximal map soft-thresholds at $s\lambda_1$ and
then divides by $1 + s\lambda_2$. Fits support [proximal
gradient](../solvers/proximal-gradient.md) and [IRLS](../solvers/irls.md).

API:
[`shrinkage::ElasticNet`](https://shrinkage.cc/packages/rust/reference/7641c13ebf6510be067e5a5c045b16f0f42f3cd34ed29cbab52422e9f76ccdb2).
