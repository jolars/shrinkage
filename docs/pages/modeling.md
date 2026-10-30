# Modeling conventions

Build a model by choosing a [datafit](datafits/index.md), a
[penalty](penalties/index.md), and a compatible
[solver](reference/solvers/index.md). The [model pages](models/index.md) put
these choices together for linear, logistic, and multivariate regression. All
components described here are available through Rust. Python exposes Gaussian
lasso and compositional Gaussian and binomial fits with the three MVP penalties,
using explicit hybrid or proximal-gradient solvers. See the [Python installation
guide](installation.md#python) and [package
guide](../../packages/python/README.md) for fitting and prediction.

## Objective and notation {#objective-and-notation}

For a design with $n$ observations and $p$ features, Shrinkage minimizes the
composite primal objective

$$
  \min_{\beta_0,\beta} P(\beta_0,\beta),
  \qquad
  P(\beta_0,\beta) = F(\beta_0,\beta) + R(\beta),
  \qquad
  \eta = \beta_0\mathbf{1} + \widetilde X\beta.
$$

The datafit $F$ averages over observations. The complete penalty $R$ includes
its strengths, with no additional factor of $n$. Datafit pages also write
$F(y,\eta)$ to express the same loss directly in predictor coordinates. The
intercept $\beta_0$ is unpenalized. Set `.fit_intercept(false)` on the datafit
to omit a free intercept. The convenience `Lasso` builder provides the same
option directly.

For an observation-scale dual variable $\theta$, write $D(\theta)$ for the dual
objective. The dual problem maximizes $D$ over its feasible domain, and the
duality gap is

$$
  G(\beta_0,\beta,\theta) = P(\beta_0,\beta) - D(\theta).
$$

For a feasible dual point, weak duality gives $G \geq 0$. Solvers evaluate gaps
through stable Fenchel decompositions when available, avoiding subtraction of
nearly equal objectives. The [solver reference](reference/solvers/index.md)
describes which methods report or use this diagnostic.

Here $\beta = (\beta_1, \ldots, \beta_p)^\mathsf{T}$ denotes optimization-scale
coefficients, and $\beta_0$ denotes the intercept. Unless you select a different
normalization, Shrinkage centers training columns when fitting an intercept and
scales them by population standard deviations. Without a fitted intercept,
automatic normalization scales columns without centering.

## Normalization and predictions {#normalization-and-predictions}

Write $\widetilde X = (X - \mathbf{1}c^\mathsf{T})S^{-1}$, where $c$ contains
training centers and $S$ is diagonal with the resolved feature scales. The
reported coefficients and intercept satisfy

$$
  \begin{aligned}
    \beta_{\mathrm{original}}   & = S^{-1}\beta,                                     \\
    \beta_{0,\mathrm{original}} & = \beta_0 - c^\mathsf{T}\beta_{\mathrm{original}}, \\
    X\beta_{\mathrm{original}} + \beta_{0,\mathrm{original}}\mathbf{1}
                                & = \widetilde X\beta + \beta_0\mathbf{1}.
  \end{aligned}
$$

The fitted result owns these original-scale parameters and retains its training
preprocessing. Prediction reuses that preprocessing. Penalty geometry and
cluster labels describe $\beta$; unequal feature scales can therefore produce
unequal reported coefficients within one optimization-scale cluster.

Use `.normalize(Normalization::None)` on a problem or model builder to use the
raw design. Explicit normalization choices are independent of the intercept
option. Explicit centering without a fitted intercept can induce a fixed
original-scale intercept, which the returned fit preserves.

Rust's `Normalization::Custom { center, scale }` also accepts sample standard
deviations with `Scaling::Sd { df: 1 }`. The `Auto` and `Standardize` presets
continue to use population standard deviations (`df: 0`). For robust
normalization, combine `Centering::Median` with
`Scaling::QuantileRange(QuantileRange::default())` to subtract medians and
divide by interquartile ranges. Custom quantile bounds use probabilities, such
as `QuantileRange::new(0.1, 0.9)`. LazyMatrix computes these statistics for
dense, sparse, and file-backed input; Shrinkage retains the fitted values for
prediction and replaces computed zero scales with one.

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

## Cross-validation {#cross-validation}

Rust cross-validation fits a complete regularization path inside each training
fold. Create validated assignments with `FoldAssignments::new(labels)`, balanced
shuffled folds with `FoldAssignments::kfold(nobs, nfolds, seed)`, or stratified
binary folds with `FoldAssignments::stratified_binomial(&y, nfolds, seed)`.
Labels start at zero and must cover at least two nonempty folds. Seeded
assignments reproduce across platforms. Fractional binomial responses require
ordinary shuffled or supplied folds.

Configure `CrossValidation::new(folds)`, then call
`problem.cross_validate_path_with(&solver, penalties, &cv)` for explicit
complete penalties, or
`problem.cross_validate_regularization_path_with(&solver, &options, &cv)` for
automatic paths. `Lasso` exposes corresponding `cross_validate_path` and
`cross_validate_regularization_path` methods. Gaussian and binomial compositions
work with their existing coordinate, IRLS, proximal, and hybrid path solvers.
Dense and CSC inputs use the enabled matrix backend; custom backends implement
`RowSubset` to select raw observations.

Each fold estimates its own preprocessing and retains it throughout its path.
Cross-validation rejects supplied centers and scales so full-data preprocessing
cannot enter a training fold. Explicit candidates use the same complete penalty
in every fold, preserving independent L1 and fusion strengths, chain edge
weights, and SLOPE profiles. Automatic candidates share decreasing fractions of
each fold's training-only path anchor. The default endpoint ratio uses the
smallest training fold's dimensions and remains fixed across folds. Automatic
early stopping is disabled. A supplied `lambda_start` acts as the same fixed
external anchor in each fold; choose it independently of held-out outcomes.

The default Gaussian score is mean squared error, twice the fitting loss.
Binomial models default to mean negative log likelihood, computed stably from
held-out logits. Use `.metric(ValidationMetric::MeanAbsoluteError)` for Gaussian
absolute error or `.metric(ValidationMetric::BinomialDeviance)` for mean
binomial deviance, including the saturated-loss correction for fractional
responses. Scores exclude the penalty. Each score averages over its held-out
observations; the reported mean weights folds equally, including unequal folds.
For $K$ fold scores $s_k$, the standard error is
$\sqrt{\sum_k(s_k - \bar s)^2/[K(K - 1)]}$. It summarizes variation between
folds without accounting for overlapping training sets.

Minimum-error selection chooses the first candidate with the lowest mean loss.
Set `.selection_rule(SelectionRule::OneStandardError)` to choose the first
candidate within one standard error of the minimum-loss point. Order explicit
candidates from most to least preferred for simplicity. This caller-supplied
order defines the rule for independent strengths and profiles; Shrinkage does
not infer a scalar complexity order for these penalties. Automatic paths use
decreasing relative strength.

Selection finishes before a fresh full-data path is fitted. For automatic paths,
that refit computes its own full-data anchor and applies the selected fraction.
The result's `fit()` and `penalty()` return the selected full-data model and
complete penalty. `folds()` retains training paths, validation row indices, and
scores; `mean_scores()`, `standard_errors()`, and `relative_strengths()` expose
the validation curve. `refit_path()` retains all full-data fits for inspection.

A binomial training fold with only zeros or only ones has no finite optimum for
a free intercept and produces a fold-specific error. Disable the intercept to
fit that training response when appropriate. Any solver error or nonconverged
candidate aborts cross-validation before selection; errors retain fold context
and backend sources. See the [runnable cross-validation
example](https://github.com/jolars/shrinkage/blob/main/examples/cross_validation.rs).

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
