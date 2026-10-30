# Clustered lasso

Clustered lasso encourages sparse coefficients and equal signed effects. Every
unordered pair contributes once. Unlike SLOPE, fusion compares signed values:
equal positive and negative magnitudes do not form one fusion cluster. The
constructor takes separate L1 and fusion strengths.

$$
  P(\beta) = \lambda_1\sum_j|\beta_j| + \lambda_2\sum_{i<j}|\beta_i - \beta_j|.
$$

## Constraint region

![Feasible region for clustered lasso; the shaded set includes its
boundary.](figures/clustered-lasso.svg)

Top: $\lambda_1 = \lambda_2 = 0.5$. Bottom: pure fusion, $\lambda_1 = 0$,
$\lambda_2 = 1$, giving the unbounded strip $|\beta_1 - \beta_2| \leq 1$. See
the [plot conventions](index.md#read-the-constraint-plots) for the fixed budget
and coefficient scale.

## Use the penalty

```rust
use shrinkage::{ClusteredLasso, Penalty, ProximalPenalty};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let penalty = ClusteredLasso::new(0.5, 0.25);
    let coefficients = [2.0, -1.0];
    println!("Penalty: {}", penalty.value(&coefficients)?);
    let mut shrunk = [0.0; 2];
    penalty.prox(&coefficients, 1.0, &mut shrunk)?;
    println!("Proximal result: {shrunk:?}");
    Ok(())
}
```

At zero fusion strength, this reduces to lasso. Without the L1 term, a common
shift of all coefficients has no penalty, so the constraint region is unbounded
along the equality direction. In two dimensions, its geometry equals that of
unit-weight chain fused lasso. With more features, all-pairs and adjacent fusion
differ. Gaussian and binomial fits support [proximal
gradient](../reference/solvers/proximal-gradient.md) and
[hybrid](../reference/solvers/hybrid.md). See the [clustered lasso
guide](../models/clustered-lasso.md) for cluster labels and logistic fitting
controls.

API:
[`shrinkage::ClusteredLasso`](https://shrinkage.cc/packages/rust/reference/22d366ee2ce4c0fc6b3806b8d650b550a59a050150c6e013fd744f04d3754924).
