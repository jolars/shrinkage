# Ridge

Ridge shrinks coefficients continuously and usually leaves them nonzero. Its
constraint region is a disk. This is a squared L2 penalty, which differs from
the unsquared norm used by Group LASSO.

$$
  P(\theta) = \frac{\lambda}{2}\|\theta\|_2^2.
$$

## Constraint region

![Feasible region for ridge; the shaded set includes its
boundary.](figures/ridge.svg)

$\lambda = 2$: $\theta_1^2 + \theta_2^2 \leq 1$. See the [plot
conventions](index.md#read-the-constraint-plots) for the fixed budget and
coefficient scale.

## Use the penalty

```rust
use shrinkage::{Penalty, ProximalPenalty, Ridge};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let penalty = Ridge::new(0.5);
    let coefficients = [2.0, -1.0];
    println!("Penalty: {}", penalty.value(&coefficients)?);
    let mut shrunk = [0.0; 2];
    penalty.prox(&coefficients, 1.0, &mut shrunk)?;
    println!("Proximal result: {shrunk:?}");
    Ok(())
}
```

Use [proximal gradient](../reference/solvers/proximal-gradient.md) or
[IRLS](../reference/solvers/irls.md) with Gaussian or binomial datafits. Its
proximal map divides each input by $1 + s\lambda$. With zero strength, it
returns the input unchanged.

API:
[`shrinkage::Ridge`](https://shrinkage.cc/packages/rust/reference/cbb3b47b3b63b801639d1697f8ab9090627986cae975bb7f06d11764acdeee8c).
