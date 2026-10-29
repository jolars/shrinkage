# Hybrid

`Hybrid` alternates proximal-gradient transitions with exact updates of
coefficient blocks. Gaussian problems use the original quadratic loss. Binomial
problems use weighted hybrid subproblems inside IRLS, with backtracking on the
original objective.

Three penalties are supported. Clustered lasso groups equal signed effects,
chain fused lasso groups contiguous equal segments, and SLOPE groups equal
magnitudes while retaining individual signs. PG transitions can split blocks and
reactivate coordinates that block sweeps leave fixed.

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

The default schedule starts with one PG transition on the working set, then
alternates five block sweeps with another PG transition. The Gaussian budget is
10,000 PG transitions and complete sweeps combined. Checks run initially, every
ten iterations, at restricted convergence, and at the budget. A changed working
set can trigger an earlier PG transition. Set `.cluster_sweeps(...)`,
`.check_every(...)`, and `.max_iterations(...)` to change these policies.

Only the full PG mapping on the original loss certifies convergence, including
the free-intercept derivative. The default absolute threshold is `1e-6`.
Unchanged block sweeps are insufficient. Backtracking is the default curvature
policy; `HybridStep::FixedBound` uses a conservative computed bound.

Logistic fitting has separate outer and inner budgets. The [logistic
controls](../../models/clustered-lasso.md#logistic-regression) explain their
meanings and the returned diagnostics. The [SLOPE](../../models/slope.md) and
[chain fused lasso](../../models/chain-fused-lasso.md) guides cover their block
semantics.

Typed Gaussian and binomial problems are supported; runtime hybrid fits and
hybrid paths are unavailable. Dense aggregates require $O(nK + p + n)$ storage
for $K$ blocks. Rebuilds stage a second aggregate set, and buffers retain their
high-water capacities. At one block per feature, each aggregate set approaches
the size of a dense design, even when the input matrix is sparse.

## Working sets

[Working sets](index.md#working-sets) are enabled independently of screening.
The default adaptive policy starts with ten features and ranks candidates by
components of the full PG mapping. Current nonzeros remain eligible. Restricted
PG transitions can split clusters and reactivate selected zero coefficients. If
excluded coefficients must enter together, a stalled restricted solve grows the
minimum feature budget until progress resumes.

Exact restrictions retain clustered-lasso interactions with omitted zeros, chain
boundaries, and SLOPE rank weights. A heuristic exclusion never establishes a
zero certificate. Only an unrestricted original-loss mapping accepts
convergence.

## Dynamic zero screening

Gap Safe zero screening is enabled by default for Gaussian and binomial fits
with clustered lasso, chain fused lasso, and SLOPE. Set `.screening(false)` to
compare with an unscreened solve. Screening fixes only certified zeros and
preserves the complete penalty, including fusion edges touching removed zeros.
Returned coefficients keep their original feature order and dimension.

`dual_certificate()`, `dual_objective()`, and `duality_gap()` return optional
original-problem diagnostics. `screening_diagnostics()` reports certificate
passes, zero tests, certified features, remaining dimensions, and rebuilds. The
full original-loss PG mapping still controls convergence. At zero sparsity
strength, a feasible zero certificate can give a loose gap and screening leaves
features eligible. Fusion screening remains deferred.

Certificates run at Gaussian checks and accepted logistic outer iterates.
Quadratic IRLS subproblem gaps never certify original logistic restrictions.

API:
[`shrinkage::Hybrid`](https://shrinkage.cc/packages/rust/reference/2a1d90805911db329d5e4ff4986bb2f7fdb00992ed6828b9cd3b3af05e3b1973),
[`shrinkage::HybridStep`](https://shrinkage.cc/packages/rust/reference/8c2324572d804bcaf844c43f1cf5a8cfa03c0ef3439c88e54a8dcdd948cdc85e),
and
[`shrinkage::HybridFit`](https://shrinkage.cc/packages/rust/reference/4cfe41c15bbeb98348e19c60ae6240bfb650cf4d7fe4c2c32a660c298508b022).
