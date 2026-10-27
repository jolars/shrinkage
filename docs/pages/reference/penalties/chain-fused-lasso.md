# Chain fused lasso

Chain fused lasso encourages sparse, piecewise constant coefficients in the
supplied feature order. Adjacent equal effects form contiguous segments. Weights
default to one. `.with_weights(...)` takes exactly one finite, nonnegative
weight per edge; a zero weight cuts the chain.

$$
  P(\theta) = \lambda_1\sum_{j=1}^p|\theta_j|
              + \lambda_2\sum_{j=1}^{p-1}w_j|\theta_{j+1} - \theta_j|.
$$

## Constraint region

![Feasible region for chain fused lasso; the shaded set includes its
boundary.](figures/chain-fused-lasso.svg)

Top: $\lambda_1 = \lambda_2 = 0.5$, $w_1 = 1$. Bottom: $\lambda_1 = 0$,
$\lambda_2 = 1$, $w_1 = 1$, giving an unbounded pure-fusion strip. See the [plot
conventions](index.md#read-the-constraint-plots) for the fixed budget and
coefficient scale.

## Use the penalty

```rust
use shrinkage::{ChainFusedLasso, Penalty, ProximalPenalty};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let penalty = ChainFusedLasso::new(0.5, 0.25).with_weights(vec![1.0]);
    let coefficients = [2.0, -1.0];
    println!("Penalty: {}", penalty.value(&coefficients)?);
    let mut shrunk = [0.0; 2];
    penalty.prox(&coefficients, 1.0, &mut shrunk)?;
    println!("Proximal result: {shrunk:?}");
    Ok(())
}
```

Each edge contributes once. Empty and single-feature chains have no edges. Zero
fusion recovers lasso. A two-feature chain has one pair, so its unit-weight
constraint region coincides with clustered lasso; larger chains only fuse
neighbors. Gaussian and binomial fits support [proximal
gradient](../solvers/proximal-gradient.md) and [hybrid](../solvers/hybrid.md).
See the [chain fused lasso guide](../../chain-fused-lasso.md) for segment
interpretation and detailed fitting controls.

API:
[`shrinkage::ChainFusedLasso`](https://shrinkage.cc/packages/rust/reference/b88554a6c3f1121915d157c3e7684558bb6e9e9f59dbacd8115eb616c3631ce1).
