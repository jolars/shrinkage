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
  \min_{b,\theta}\; L(y,\eta) + P(\theta),
  \qquad
  \eta = b\mathbf{1} + \widetilde X\theta.
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
  \beta = S^{-1}\theta,\qquad b_{\mathrm{original}} = b - c^\mathsf{T}\beta,
  \qquad
  X\beta + b_{\mathrm{original}}\mathbf{1} = \widetilde X\theta + b\mathbf{1}.
$$

The fitted result owns these original-scale parameters and retains its training
preprocessing. Prediction reuses that preprocessing. Penalty geometry and
cluster labels describe $\theta$; unequal feature scales can therefore produce
unequal reported coefficients within one optimization-scale cluster.

Use `.normalize(Normalization::None)` on a problem or model builder to use the
raw design. Explicit normalization choices are independent of the intercept
option. Explicit centering without a fitted intercept can induce a fixed
original-scale intercept, which the returned fit preserves.

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
