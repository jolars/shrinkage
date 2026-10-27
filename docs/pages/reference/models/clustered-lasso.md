# Clustered lasso

Clustered lasso encourages sparse coefficients and equal signed effects. The
Rust API provides its exact proximal map, Gaussian proximal-gradient fits, and a
hybrid solver that interleaves PG steps with exact cluster-coordinate sweeps.

See the [penalty reference](../penalties/clustered-lasso.md) for its definition
and constraint geometry.

## Fit a Gaussian model

Enable the matrix integration you use, such as `faer_v0_24`, then compose the
design, loss, and penalty:

```rust
use faer::{Col, Mat};
use shrinkage::{ClusteredLasso, Gaussian, Hybrid, Problem, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(8, 3, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
    let y: Vec<_> = (0..8)
        .map(|i| 2.0 + x[(i, 0)] + x[(i, 1)] - 0.5 * x[(i, 2)])
        .collect();
    let problem = Problem::new(&x, Gaussian::new(&y), ClusteredLasso::new(0.05, 0.1));
    let fit = problem.fit_with(&Hybrid::<Col<f64>>::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Cluster labels: {:?}", fit.cluster_labels());
    println!("Cluster values: {:?}", fit.cluster_values());
    println!("Predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
```

The objective uses averaged squared loss and separate nonnegative strengths:

$$
  \frac{\|y - b - \widetilde X\theta\|^2}{2n}
  + \lambda_1\sum_j|\theta_j| + \lambda_2\sum_{i<j}|\theta_i - \theta_j|.
$$

Every unordered pair contributes once. Fusion compares signed values, so
opposite signs remain distinct. The intercept is unpenalized. As with lasso,
automatic normalization standardizes training columns and centers them when an
intercept is fitted. Returned coefficients and predictions use the original
scale.

Cluster labels index the strictly increasing optimization-scale values in
`fit.cluster_values()`. Unequal training scales can produce unequal reported
coefficients inside one optimization block. The retained preprocessing explains
that transformation.

## Choose a solver and accuracy

The hybrid takes a full PG step first, then performs five sweeps between full PG
steps. Sweeps can move zero blocks, cross other blocks, and merge exact ties.
Full PG steps can split blocks. Only a fresh full-space PG mapping establishes
convergence; unchanged sweeps do not.

Defaults are an absolute mapping tolerance of `1e-6`, 10,000 iterations, and a
check every ten iterations, with initial and final checks. One iteration is one
PG transition or one complete sweep. Configure these through `.tolerance(...)`,
`.max_iterations(...)`, `.cluster_sweeps(...)`, and `.check_every(...)`. The
mapping also checks the unpenalized intercept derivative. Duality-gap and KKT
stopping selections are unsupported for this hybrid.

Backtracking is the default. To use a computed conservative curvature bound:

```rust
use shrinkage::{Hybrid, HybridStep};

let solver = Hybrid::<Vec<f64>>::new()
    .step_policy(HybridStep::FixedBound)
    .tolerance(1e-8);
```

For explicit backtracking controls, select
`HybridStep::Backtracking { initial_curvature: Some(1.0), max_backtracks: 100 }`.
The supplied curvature is a positive seed, and accepted curvature can increase.

Use `ProximalGradient::<Col<f64>>::new()` with the same problem for the proximal
baseline. The penalty also composes with runtime PG through `RuntimeProblem`.
The hybrid uses typed Gaussian or binomial problems and dense aggregate storage.
Its result reports `pg_steps()`, `coordinate_sweeps()`, `diagnostic_checks()`,
`backtracks()`, and the final objective, step, and mapping. Inspect termination:
an iteration limit returns finite parameters without certifying convergence.

Aggregate memory grows with the number of clusters. Each retained aggregate set
uses approximately `8 * n * K` bytes, reaching the size of a dense design when
every feature has its own block. PG transitions stage a second set, and buffers
retain their high-water capacities. The solver does not construct a Gram matrix
or an explicit pairwise graph.

The heap benchmark measured these peak solver allocations on 64-bit Linux with
Rust 1.89.0. It excludes the caller's input arrays and includes setup, retained
results, and final diagnostics. The one-block fits converge; the distinct-block
cases run seven iterations of an unpenalized fit to expose aggregate storage.

  | Observations | Features | One block | One block per feature |
  | ------------ | -------- | --------- | --------------------- |
  | 128          | 64       | 22,504 B  | 169,504 B             |
  | 512          | 64       | 50,152 B  | 584,224 B             |
  | 128          | 256      | 59,368 B  | 656,416 B             |

Warmed value and prox calls allocate no additional storage at a fixed dimension.
Complete solves still allocate their workspaces and fitted results.

## Reproduce the research convention

The reference revision is `293760ea4040351a654906add5d7ca855be1deb9` from the
clustered-lasso research project. Its squared loss is unaveraged. Use
`Gaussian::new(&y).fit_intercept(false)`, `Normalization::None`, and strengths
`research_l1 / n` and `research_fusion / n` to obtain the same minimizers.
Corresponding curvature bounds divide by `n`, and objectives divide by `n`.

Research stopping normalizes its mapping by `max(1, ||X' y||_inf)`. Convert its
tolerance to Shrinkage's absolute mapping units by multiplying by that scale and
dividing by `n`. The complete derivation, scalar updates, PG transition
contracts, convergence assumptions, and source hashes are retained in the
[design document](https://github.com/jolars/shrinkage/blob/main/DESIGN.md).

The Rust tests consume committed independent epigraph-QP fixtures. Regenerate
them from a checkout with `uv run --script scripts/clustered_lasso_fixtures.py`;
the generator pins NumPy, CVXPY, and Clarabel and imports neither Shrinkage nor
the research package. Run the example and benchmarks with:

```console
devenv shell -- cargo run --locked --example clustered_lasso --features faer_v0_24
devenv shell -- task bench:clustered
devenv shell -- task bench:clustered:heap
```

## Logistic regression {#logistic-regression}

Use `Binomial::new(&y)` with the same `Hybrid<V>` solver to fit logistic
regression. Clustered lasso, chain fused lasso, and SLOPE all use weighted
hybrid subproblems inside IRLS. For example:

```rust
use faer::{Col, Mat};
use shrinkage::{Binomial, ClusteredLasso, Hybrid, Problem, Termination};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let x = Mat::from_fn(8, 2, |i, j| ((i + j) % 3) as f64);
let y = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
let fit = Problem::new(&x, Binomial::new(&y), ClusteredLasso::new(0.02, 0.03))
    .fit_with(&Hybrid::<Col<f64>>::new())?;
assert_eq!(fit.termination(), Termination::Converged);
let logits = fit.predict(&x)?;
# Ok(())
# }
```

The default limits are 100 accepted IRLS steps, 1,000 inner PG transitions or
sweeps per quadratic model, and 10,000 inner iterations across the whole fit.
Configure them with `max_outer_iterations`, `max_inner_iterations`, and
`max_iterations`. `max_outer_backtracks` limits logistic step halvings, while
`HybridStep` controls quadratic PG curvature. The final objective and mapping
use the original logistic loss. Check `termination()` before interpreting a fit
as converged. `outer_iterations()` and `outer_backtracks()` report outer work;
the existing PG and sweep counters report inner work. Predictions are logits.

Run `task bench:hybrid:irls` for complete-fit comparisons with PG and
`task bench:hybrid:irls:heap` for weighted workspace storage checks.

Python exposure, hybrid paths, screening, and sparse-specific extensions follow
in later roadmap milestones.
