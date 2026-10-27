# Hybrid

`Hybrid` alternates full proximal-gradient transitions with exact updates of
coefficient blocks. Gaussian problems use the original quadratic loss. Binomial
problems use weighted hybrid subproblems inside IRLS, with backtracking on the
original objective.

Three penalties are supported. Clustered lasso groups equal signed effects,
chain fused lasso groups contiguous equal segments, and SLOPE groups equal
magnitudes while retaining individual signs. Full PG transitions can split
blocks and reactivate coordinates that block sweeps leave fixed.

## Fit clustered lasso

```rust
use faer::{Col, Mat};
use shrinkage::{ClusteredLasso, Gaussian, Hybrid, Problem, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(8, 3, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
    let y: Vec<_> = (0..8).map(|i| 2.0 + x[(i, 0)] + x[(i, 1)]).collect();
    let fit = Problem::new(&x, Gaussian::new(&y), ClusteredLasso::new(0.05, 0.1))
        .fit_with(&Hybrid::<Col<f64>>::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Cluster labels: {:?}", fit.cluster_labels());
    println!("Cluster values: {:?}", fit.cluster_values());
    Ok(())
}
```

## Controls and convergence

The default schedule starts with one full PG transition, then alternates five
complete block sweeps with another PG transition. The Gaussian budget is 10,000
PG transitions and complete sweeps combined. Checks run initially, every ten
iterations, and at the budget. Set `.cluster_sweeps(...)`, `.check_every(...)`,
and `.max_iterations(...)` to change these policies.

Only the full PG mapping on the original loss certifies convergence, including
the free-intercept derivative. The default absolute threshold is `1e-6`.
Unchanged block sweeps are insufficient. Backtracking is the default curvature
policy; `HybridStep::FixedBound` uses a conservative computed bound.

Logistic fitting has separate outer and inner budgets. The [logistic
controls](../../clustered-lasso.md#logistic-regression) explain their meanings
and the returned diagnostics. The [SLOPE](../../slope.md) and [chain fused
lasso](../../chain-fused-lasso.md) guides cover their block semantics.

Typed Gaussian and binomial problems are supported; runtime hybrid fits and
hybrid paths are unavailable. Dense aggregates require $O(nK + p + n)$ storage
for $K$ blocks. Rebuilds stage a second aggregate set, and buffers retain their
high-water capacities. At one block per feature, each aggregate set approaches
the size of a dense design, even when the input matrix is sparse.

API:
[`shrinkage::Hybrid`](https://shrinkage.cc/packages/rust/reference/2a1d90805911db329d5e4ff4986bb2f7fdb00992ed6828b9cd3b3af05e3b1973),
[`shrinkage::HybridStep`](https://shrinkage.cc/packages/rust/reference/8c2324572d804bcaf844c43f1cf5a8cfa03c0ef3439c88e54a8dcdd948cdc85e),
and
[`shrinkage::HybridFit`](https://shrinkage.cc/packages/rust/reference/4cfe41c15bbeb98348e19c60ae6240bfb650cf4d7fe4c2c32a660c298508b022).
