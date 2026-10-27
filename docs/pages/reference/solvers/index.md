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
[installation](../../installation.md) and the [shared conventions](../index.md).

API:
[`shrinkage::Solver`](https://shrinkage.cc/packages/rust/reference/f80ebf14b213d3129f84bb1ccc4912d7d7123fed16e872d24da1a9f5ef741b0a),
[`shrinkage::StoppingCriterion`](https://shrinkage.cc/packages/rust/reference/3b62080d01f42c76b09853b75250a14941119801c6f676b199a1a311bb65cdcb),
and
[`shrinkage::PathOptions`](https://shrinkage.cc/packages/rust/reference/21f99e48696dcba0a8b440da1f7edf02d640beefa18dad50b5605b9a2811fff1).
