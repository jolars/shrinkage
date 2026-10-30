# SLOPE

SLOPE assigns larger penalties to larger coefficient magnitudes. Shrinkage
provides its exact proximal map, Gaussian proximal-gradient fits, and a hybrid
of full PG steps and exact magnitude-cluster sweeps.

See the [penalty guide](../penalties/slope.md) for its definition and constraint
geometry.

## Fit a Gaussian model

Enable your matrix integration, such as `faer_v0_24`, and supply one finite,
nonnegative, nonincreasing weight per feature, including the penalty strength:

```rust
use faer::{Col, Mat};
use shrinkage::{Gaussian, Hybrid, Problem, Slope, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(16, 5, |i, j| ((i * 7 + j * 3 + i * j) % 17) as f64 - 8.0);
    let y: Vec<_> = (0..16)
        .map(|i| 2.0 + x[(i, 0)] - x[(i, 1)] + 0.5 * x[(i, 3)] - 0.5 * x[(i, 4)])
        .collect();
    let penalty = Slope::new(vec![0.5, 0.4, 0.3, 0.2, 0.1]);
    let fit = Problem::new(&x, Gaussian::new(&y), penalty).fit_with(&Hybrid::<Col<f64>>::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Magnitude labels: {:?}", fit.cluster_labels());
    println!("Cluster magnitudes: {:?}", fit.cluster_values());
    println!("Individual signs: {:?}", fit.coefficient_signs());
    Ok(())
}
```

The complete objective is

$$
  \frac{\|y - \beta_0\mathbf{1} - \widetilde X\beta\|_2^2}{2n}
  + \sum_{j=1}^p w_j|\beta|_{(j)}.
$$

Here $|\beta|_{(j)}$ is the $j$th largest magnitude on the optimization scale.
The supplied weights include the strength, with no additional factor of `n`.
Equal weights give lasso; all zero weights give an unpenalized fit. Repeated
weights and trailing zeros are valid. Empty feature vectors require empty
weights. Intercepts are unpenalized.

## Interpret magnitude clusters

A SLOPE block contains exactly equal magnitudes and can include positive and
negative coefficients. `cluster_values()` returns nonnegative magnitudes in
strictly decreasing order, including a final zero block when present.
`cluster_labels()` indexes those values. Recover optimization coefficient `j` as
`coefficient_signs()[j] * cluster_values()[cluster_labels()[j]]`. Both signed
zeros have sign zero.

The default preprocessing standardizes columns and centers them when fitting an
intercept. Returned coefficients and predictions use the original input scale.
Clusters describe the optimization scale: unequal training scales can make
members of one magnitude cluster have different original-scale magnitudes.
Back-transformation preserves predictions. Select `Normalization::None` and
disable the intercept to fit an unnormalized, intercept-free objective.

## Control and inspect fitting

The hybrid starts with a PG transition on the working set, then alternates five
cluster sweeps with another PG by default. Exact scalar updates can cross ranks,
flip every sign in a cluster, and merge exact magnitude ties. The zero block
stays fixed during sweeps. Full PG transitions rebuild the partition, split
clusters, and reactivate zero coordinates.

Use `.cluster_sweeps(count)`, `.max_iterations(count)`, and
`.check_every(count)` to control work. One iteration is one committed PG
transition or complete sweep. The default budget is 10,000 iterations, with
checks initially, every ten iterations, and at the budget. Unchanged sweeps do
not establish convergence.

`.tolerance(value)` sets the absolute infinity-norm full PG mapping threshold,
defaulting to `1e-6`. The check includes the free-intercept derivative and uses
fresh residuals at the returned parameters. Inspect `termination()` before
treating a fit as converged; finite budget exhaustion reports `IterationLimit`.
Diagnostics report the objective, stopping value and threshold, PG steps,
coordinate sweeps, checks, backtracks, and step size.

The default backtracking policy checks the averaged Gaussian quadratic model and
retains accepted curvature. `HybridStep::FixedBound` uses a computed Frobenius
upper bound. Both include the intercept when enabled. No Gram matrix or
normalized design is materialized. Each partition stores dense aggregates using
`O(n K_max + p + n)` memory, with buffers retained at the high-water cluster
count. PG rebuild staging retains two partition states. At `K_max=p`, each
aggregate set is about the size of another dense design.

## Reproduce validation

Independent top-k epigraph QPs and subgradient certificates verify Gaussian
fits. Scalar tests enumerate signed intervals, and sweep tests rebuild every
restricted problem. The [design
notes](https://github.com/jolars/shrinkage/blob/main/DESIGN.md) describe the
rank-dependent scalar restriction and numerical conventions.

```console
devenv shell -- uv run --script scripts/slope_fixtures.py
devenv shell -- cargo run --locked --example slope --features faer_v0_24
devenv shell -- task bench:slope
devenv shell -- task bench:slope:hybrid
devenv shell -- task bench:slope:heap
```

The complete-solve benchmarks construct known distinct, tied, and highly
clustered optima and verify convergence at matched accuracy before timing PG and
both hybrid policies. Setup, rebuilds, and final diagnostics are included.
Additional bounded runs measure setup through the first PG and through the
second PG after five sweeps. Heap checks cover low, high, and changing cluster
counts and warmed proximal buffers. Results depend on the design and solution
structure; solver selection remains explicit.

A local release-build run on October 26, 2026, on an AMD Ryzen 9 7900 gave the
following point estimates for 256 observations and 64 features, triangular
correlation `0.8`, and mapping tolerance `1e-8`. Times are milliseconds. Each
fit includes preprocessing, workspace allocation, rebuilds, and final checks;
design, response, and penalty construction are outside the timed region.

  | Solution structure     | PG    | Hybrid, backtracking | Hybrid, fixed bound |
  | ---------------------- | ----: | -------------------: | ------------------: |
  | 64 distinct magnitudes | 43.03 |                22.23 |               23.68 |
  | 16 tied magnitudes     | 41.98 |                 5.96 |                6.03 |
  | One magnitude          | 33.94 |                0.276 |               0.219 |

For the orthogonal 512-by-256 heap cases, complete-fit peak live storage was
89,032 bytes with one magnitude cluster and 2,252,800 bytes with 256 clusters.
These measurements include solver workspaces and returned results, excluding the
previously constructed inputs. The warmed proximal checks allocated no
additional blocks.

Typed Gaussian and binomial fits support the hybrid through compatible matrix
adapters. Replace `Gaussian::new(&y)` with `Binomial::new(&y)` for logistic
regression through hybrid IRLS. See the [logistic fitting
controls](clustered-lasso.md) for budgets, diagnostics, and benchmarks.
Predictions remain logits.

Runtime problems support SLOPE through proximal gradient. Gaussian and binomial
[hybrid paths](../modeling.md#regularization-paths) scale the supplied weight
profile. Python exposes single fits and prediction with explicit hybrid or PG
selection; see the [Python package guide](../../../packages/python/README.md).
Python paths, fusion screening, and sparse-specific optimization follow later
roadmap milestones.
