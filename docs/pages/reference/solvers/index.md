# Solvers

Select a solver explicitly after choosing the datafit and penalty. Typed
compositions require an implemented `Solver` capability; unsupported typed
combinations fail to compile. Runtime selections perform capability checks and
return an unsupported-combination error when necessary.

  | Solver                                      | Datafit               | Penalties                                        | Paths                                                        | Default stopping check                              |
  | ------------------------------------------- | --------------------- | ------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- |
  | [Coordinate descent](coordinate-descent.md) | Gaussian              | L1                                               | Explicit and automatic                                       | Relative duality gap; absolute KKT at zero strength |
  | [Proximal gradient](proximal-gradient.md)   | Gaussian and binomial | All seven base penalties and compatible wrappers | Explicit; automatic when the penalty provides PathPenalty    | Absolute full PG mapping                            |
  | [Proximal gradient](proximal-gradient.md)   | Multivariate Gaussian | AxisPenalty with an exact inner prox             | Explicit; automatic with a compatible inner path penalty     | Absolute full PG mapping                            |
  | [IRLS](irls.md)                             | Gaussian and binomial | L1, ridge, elastic net                           | Explicit and automatic when the penalty provides PathPenalty | Absolute original-loss KKT                          |
  | [Hybrid](hybrid.md)                         | Gaussian and binomial | Clustered lasso, chain fused lasso, SLOPE        | Not implemented                                              | Absolute original-loss full PG mapping              |

Coordinate descent and proximal gradient support runtime problems. IRLS and
hybrid use typed problems; selecting them on an erased runtime problem returns
an error. Proximal gradient also supports runtime multivariate problems.

## Interpret convergence

Tolerances have units specific to the selected criterion. A lasso duality-gap
check compares the primal-dual gap with
$\epsilon_{\mathrm{abs}} + \epsilon_{\mathrm{rel}}L_0$, where $L_0$ is the fixed
zero-coefficient reference loss. KKT violation and PG mapping checks use an
absolute infinity-norm threshold on the optimization scale. All default
numerical tolerances are `1e-6` in their respective units.

The PG mapping for the penalized coefficients is

$$
  G_s(\theta)
  = \frac{\theta - \operatorname{prox}_{sP}(\theta - s\nabla_\theta L)}{s}.
$$

With a fitted intercept, the full check also includes the unpenalized intercept
derivative. Solvers refresh the state used for final checks. Iteration limits
remain independent work budgets and report `IterationLimit`, not convergence.
Check frequencies and iteration units depend on the solver.

## Working sets {#working-sets}

`CoordinateDescent`, `Irls`, `Hybrid`, and the `Lasso` builder enable working
sets by default. `WorkingSetPolicy::Auto` uses adaptive sets for single fits and
`EverActive` for existing paths. Override it with `.working_set_policy(...)`, or
use `.working_sets(false)` to disable heuristic restrictions. Calling
`.working_sets(true)` restores `Auto`; the last setter wins.

```rust
use shrinkage::{CoordinateDescent, WorkingSetPolicy};

let solver = CoordinateDescent::new().working_set_policy(
    WorkingSetPolicy::Adaptive { initial_size: 10, prune: true },
);
```

`EverActive` retains features that become nonzero at accepted iterates,
including previous path points. It discovers activity with a full eligible sweep
or PG transition on a cold start. Complete checks add excluded optimality
violators.

`Adaptive` always retains current nonzeros and fills its budget with the largest
optimality violations, breaking ties by feature index. Separable penalties use
coordinate KKT violations; Hybrid uses full PG mapping components. With
`prune: false`, unsuccessful restricted solves retain and double the set. With
`prune: true`, they rebuild around twice the current support, with the initial
size as a floor. Stalled solves increase a persistent minimum budget, eventually
allowing the full problem. The default initial size is ten; it must be positive.
This combines Celer-inspired growth with general optimality scoring rather than
Celer's dual-distance ranking.

Full checks refresh excluded features, and only original-loss checks certify
convergence. IRLS rebuilds selection for each quadratic model while retaining
activation history. Path points reset certificates and scores, retaining
EverActive history even for increasing or repeated strengths. Screening controls
remain independent of working-set policy.

`working_set_diagnostics()` reports the resolved policy, full and restricted
checks, rebuilds, budget expansions, feature reactivations, eligible dimensions,
and feature and block visits. Counts refer to one fitted model, including its
quadratic subproblems. `task bench:working-sets` compares complete fits and
paths, including starting sizes 10, 100, and 200, at independently checked
accuracy.

## Paths and storage

`Problem::fit_path_with` accepts explicit penalties in caller order and reuses
solver storage. `fit_regularization_path_with` additionally requires a
penalty-specific maximum-strength and scaling capability. Built-in automatic
paths support L1, ridge, elastic net, SLOPE, and compatible axis penalties.
Ridge has no finite zero-coefficient threshold for a nonzero null gradient; its
default starting anchor is the maximum absolute null gradient divided by
`0.001`. Elastic net with zero L1 mixing uses the same ridge convention. SLOPE
paths multiply the supplied rank weights by a scalar strength. Fusion, group,
scaled, and masked penalties do not currently provide automatic paths.

No solver forms a Gram matrix or materializes the normalized design by default.
Hybrid solvers retain dense cluster aggregates, which can grow to the size of
the dense design. Matrix backend requirements remain solver-specific; consult
[installation](../../installation.md) and the [shared
conventions](../../modeling.md).

API:
[`shrinkage::Solver`](https://shrinkage.cc/packages/rust/reference/f80ebf14b213d3129f84bb1ccc4912d7d7123fed16e872d24da1a9f5ef741b0a),
[`shrinkage::StoppingCriterion`](https://shrinkage.cc/packages/rust/reference/3b62080d01f42c76b09853b75250a14941119801c6f676b199a1a311bb65cdcb),
and
[`shrinkage::PathOptions`](https://shrinkage.cc/packages/rust/reference/21f99e48696dcba0a8b440da1f7edf02d640beefa18dad50b5605b9a2811fff1).
