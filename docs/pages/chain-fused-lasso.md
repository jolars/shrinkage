# Chain fused lasso

Chain fused lasso encourages sparse coefficients and equal effects on adjacent
features. The Rust API provides an exact weighted proximal map, Gaussian
proximal-gradient fits, and a hybrid of full PG steps and exact segment sweeps.

## Fit a Gaussian model

Enable your matrix integration, such as `faer_v0_24`, and compose the design,
loss, and penalty:

```rust
use faer::{Col, Mat};
use shrinkage::{ChainFusedLasso, Gaussian, Hybrid, Problem, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(16, 5, |i, j| ((i * 7 + j * 3 + i * j) % 17) as f64 - 8.0);
    let y: Vec<_> = (0..16)
        .map(|i| 2.0 + x[(i, 0)] + x[(i, 1)] - 0.5 * x[(i, 3)] - 0.5 * x[(i, 4)])
        .collect();
    let penalty = ChainFusedLasso::new(0.05, 0.2).with_weights(vec![1.0, 0.5, 0.0, 2.0]);
    let fit = Problem::new(&x, Gaussian::new(&y), penalty).fit_with(&Hybrid::<Col<f64>>::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Segment labels: {:?}", fit.cluster_labels());
    println!("Segment values: {:?}", fit.cluster_values());
    Ok(())
}
```

The objective is

```text
||y - b - X_tilde theta||² / (2n)
  + l1 * sum_j |theta[j]|
  + fusion * sum_j weights[j] * |theta[j+1] - theta[j]|
```

Both strengths are finite and nonnegative. Unit weights are implicit unless
`.with_weights(...)` supplies one finite, nonnegative weight per adjacent edge.
Each edge contributes once, following training-column order. A zero weight
removes that fusion term and cuts the chain. Empty and single-feature chains
have no edges. Dimensions and weights are checked even at zero strength.

The intercept is unpenalized. Automatic normalization standardizes training
columns and centers them when an intercept is fitted. The penalty acts on
optimization-scale coefficients; returned parameters use the original scale, and
back-transformation preserves predictions.

## Read segments and convergence

Segments contain contiguous, exactly equal signed optimization-scale
coefficients. Equal values separated by another segment remain separate, and
zero-weight edges always remain boundaries. Signed zeros are normalized.
`fit.cluster_labels()` indexes `fit.cluster_values()` in feature order. These
values may repeat and need not increase. Unequal column scales can produce
unequal reported coefficients within one optimization-scale segment.

The hybrid begins with a full PG step, then performs five segment sweeps between
PG steps. Sweeps move whole segments, permit crossings, and merge adjacent exact
ties. Full PG steps rebuild the partition and allow splitting. Convergence
requires a fresh full-space PG mapping, including the free-intercept derivative;
unchanged sweeps cannot certify convergence.

Defaults are an absolute mapping tolerance of `1e-6`, a budget of 10,000 PG
steps and complete sweeps combined, and checks initially, every ten iterations,
and at the budget. Configure `.tolerance(...)`, `.max_iterations(...)`,
`.cluster_sweeps(...)`, and `.check_every(...)`. Backtracking is the default;
`.step_policy(HybridStep::FixedBound)` uses a conservative computed bound. The
fit reports the objective, mapping, termination reason, PG steps, sweeps,
diagnostic checks, and backtracks. An iteration limit does not certify
convergence. No chain-specific duality gap is implemented.

Use `ProximalGradient::<Col<f64>>::new()` on the same problem for the reference
solver. The complete prox applies weighted taut strings followed by uniform L1
soft-thresholding in `O(p)` time with reusable `O(p)` scratch. Runtime problems
support PG; hybrid fits use typed Gaussian or binomial problems and dense
segment aggregates. Each retained aggregate set costs approximately `8 * n * K`
bytes for `K` segments, and PG transitions stage a second set. Buffers retain
their high-water capacities.

## Reproduce validation

Tests use independent weighted-edge epigraph-QP fixtures and dual certificates,
including nonuniform weights, cuts, rank-deficient designs, zero-strength
limits, and intercepts. Regenerate them with the pinned generator:

```console
devenv shell -- uv run --script scripts/chain_fused_lasso_fixtures.py
devenv shell -- cargo run --locked --example chain_fused_lasso --features faer_v0_24
devenv shell -- task bench:chain
devenv shell -- task bench:chain:heap
```

Timing benchmarks include complete solves at matched accuracy and bounded runs
covering setup, the first PG step, and a subsequent rebuild after five sweeps.
Those bounded timings include preprocessing and final diagnostics. Heap checks
measure warmed proximal allocations and complete-solve storage with few segments
and nearly one segment per feature. Replace `Gaussian::new(&y)` with
`Binomial::new(&y)` to fit logistic regression through hybrid IRLS. See the
[logistic fitting controls](clustered-lasso.md) for budgets, diagnostics, and
benchmarks. Predictions remain logits.

Python exposure, hybrid paths, screening, and sparse-specific optimizations
follow later roadmap milestones.
