# Penalties

A penalty assigns a cost to optimization-scale coefficients. Intercepts are
excluded, and strengths are already included in every penalty's value. All
built-in strengths must be finite and nonnegative.

  | Penalty                                   | Encourages                                   |
  | ----------------------------------------- | -------------------------------------------- |
  | [L1](l1.md)                               | Sparse individual coefficients               |
  | [Ridge](ridge.md)                         | Small coefficients through a squared L2 cost |
  | [Elastic net](elastic-net.md)             | Sparsity together with quadratic shrinkage   |
  | [SLOPE](slope.md)                         | Sparsity and equal coefficient magnitudes    |
  | [Group LASSO](group-lasso.md)             | Sparse, explicitly defined groups            |
  | [Clustered lasso](clustered-lasso.md)     | Sparsity and equal signed effects            |
  | [Chain fused lasso](chain-fused-lasso.md) | Sparsity and adjacent equal effects          |

Use [Scaled](scaled.md) to multiply a complete penalty, [Masked](masked.md) to
exclude supported coordinates, and [AxisPenalty](axis.md) to apply a vector
penalty along coefficient-matrix axes.

## Read the constraint plots {#read-the-constraint-plots}

Each page shows the feasible region for a two-coordinate constrained problem,

$$
  \min_{\beta_0,\beta} F(\beta_0,\beta) \quad\text{subject to}\quad R(\beta)
  \leq t.
$$

The plots fix $t = 1$ and shade the feasible region, including its boundary.
Axes have equal scales and represent optimization coefficients, not reported
original-scale coefficients. Captions specify every strength, weight, and group.
The intercept does not appear in these plots.

For a penalized solution, choosing $t = R(\widehat\beta)$ makes that solution
optimal for the corresponding constrained problem. The budget $t$ is not the
penalty multiplier: the appropriate correspondence depends on the data and the
solution. Our fixed budget illustrates geometry rather than fitting a dataset.
Zero strengths and unpenalized directions can give unbounded regions; arrows
indicate continuation beyond the plotted window.

## Compose supported capabilities

[`shrinkage::Penalty`](https://shrinkage.cc/packages/rust/reference/a4f4243001e8d9121fca7a92bcb924b17aa5375aba23987fae95b2a05faf5245)
evaluates the complete cost. A compatible
[`shrinkage::ProximalPenalty`](https://shrinkage.cc/packages/rust/reference/45434a0394c5d718086e7e777f7db8271e2fb2b568e2519720168e5ea2877ed0)
also minimizes

$$
  \operatorname{prox}_{sR}(v)
  = \mathop{\arg\min}_{z}\left\{ \frac{\|z - v\|_2^2}{2s} + R(z) \right\},
  \qquad
  s > 0.
$$

All seven base penalties provide this exact complete proximal map and work with
[proximal gradient](../reference/solvers/proximal-gradient.md). A sum does not
acquire an exact map simply because its terms have individual maps. Other
solvers need additional capabilities; see the [compatibility
table](../reference/solvers/index.md).
