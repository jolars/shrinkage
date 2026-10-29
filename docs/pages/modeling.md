# Modeling conventions

Build a model by choosing a [datafit](datafits/index.md), a
[penalty](penalties/index.md), and a compatible
[solver](reference/solvers/index.md). The [model pages](models/index.md) put
these choices together for linear, logistic, and multivariate regression. All
components described here are available through Rust. Python currently exposes
Gaussian lasso; see the [Python installation guide](installation.md#python).

## Objective and notation

For a design with $n$ observations and $p$ features, Shrinkage solves

$$
  \min_{b,\theta}\; L(y,\eta)
  + P(\theta), \qquad \eta = b\mathbf{1} + \widetilde X\theta.
$$

The datafit $L$ averages over observations. The complete penalty $P$ includes
its strengths, with no additional factor of $n$. The intercept $b$ is
unpenalized. Set `.fit_intercept(false)` on the datafit to omit a free
intercept. The convenience `Lasso` builder provides the same option directly.

Here $\theta$ denotes optimization-scale coefficients. Unless you select a
different normalization, Shrinkage centers training columns when fitting an
intercept and scales them by population standard deviations. Without a fitted
intercept, automatic normalization scales columns without centering.

## Normalization and predictions {#normalization-and-predictions}

Write $\widetilde X = (X - \mathbf{1}c^\mathsf{T})S^{-1}$, where $c$ contains
training centers and $S$ is diagonal with the resolved feature scales. The
reported coefficients and intercept satisfy

$$
  \beta = S^{-1}\theta,\qquad b_{\mathrm{original}}
        = b - c^\mathsf{T}\beta, \qquad X\beta + b_{\mathrm{original}}\mathbf{1}
        = \widetilde X\theta + b\mathbf{1}.
$$

The fitted result owns these original-scale parameters and retains its training
preprocessing. Prediction reuses that preprocessing. Penalty geometry and
cluster labels describe $\theta$; unequal feature scales can therefore produce
unequal reported coefficients within one optimization-scale cluster.

Use `.normalize(Normalization::None)` on a problem or model builder to use the
raw design. Explicit normalization choices are independent of the intercept
option. Explicit centering without a fitted intercept can induce a fixed
original-scale intercept, which the returned fit preserves.

## Regularization paths {#regularization-paths}

All solver families support warm-started paths for their compatible problems.
`problem.fit_path_with(&solver, penalties)` fits complete penalties in caller
order, including repeated or increasing strengths. Every supplied point is
fitted. Hybrid supports Gaussian and binomial paths with clustered lasso, chain
fused lasso, and SLOPE. Its paths reuse preprocessing, partitions, and IRLS
buffers while refreshing screening and working sets at every point.

`problem.fit_regularization_path_with(&solver, &PathOptions::new())` generates
up to 100 decreasing, log-spaced strengths. Clustered and chain fused paths
multiply both supplied strengths by the scalar lambda, preserving their ratio
and chain edge weights. SLOPE scales its supplied rank weights. Supply explicit
complete penalties to vary the L1 and fusion strengths independently.

Automatic paths use penalty-specific anchors with intercepts optimized. Sparse
penalties start at the zero-model threshold; ridge uses a finite reference
anchor. Pure fusion can have no finite zero-model threshold; supply
`PathOptions::new().lambda_start(...)` or an explicit sequence in that case.
Automatic binomial paths with a free intercept require a finite null intercept.

By default, automatic paths stop after at least five points when the fractional
deviance improvement falls below `1e-5` or the fraction of null deviance
explained reaches `0.999`. Use `.early_stopping(false)` to fit the entire
sequence, or adjust `.min_points(...)`, `.deviance_change_tolerance(...)`, and
`.deviance_ratio_tolerance(...)`. Only converged points can trigger stopping.
Losses without deviance support fit their entire sequence.

`path.termination()` reports completion or the stopping criterion, independently
of each fit's convergence. `path.lambdas()`, `path.penalties()`, and
`path.fits()` contain the fitted prefix. Automatic paths also report
`null_deviance()` and `deviances()` when supported. Deviance is twice the loss
above the saturated loss, averaged over observations; multivariate Gaussian
deviance sums over responses. Binomial deviance includes the saturated-loss
correction for fractional responses. See the [runnable path
example](https://github.com/jolars/shrinkage/blob/main/examples/regularization_path.rs).

## Examples and diagnostics

The fitting examples use `faer::Mat` and, for proximal and hybrid solvers,
`faer::Col<f64>`. Enable `faer_v0_24` and depend on faer 0.24 as described in
[installation](installation.md). Penalty and loss evaluation examples require no
matrix backend.

Always inspect `fit.termination()`. A finite iteration budget can return useful
parameters with `Termination::IterationLimit`; that result does not certify
convergence. Solver tolerances use different units, so consult the [solver
comparison](reference/solvers/index.md) before comparing tolerances.

See the [Rust API](https://shrinkage.cc/packages/rust/) for signatures and the
[design document](https://github.com/jolars/shrinkage/blob/main/DESIGN.md) for
implementation contracts and numerical derivations.
