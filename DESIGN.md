# Shrinkage: design

Status: Gaussian lasso convenience API, typed Gaussian/L1 coordinate descent,
and typed and runtime proximal gradient with Gaussian and binomial datafits,
ridge, elastic net, SLOPE, disjoint Group LASSO, clustered lasso, chain fused
lasso, and multivariate Gaussian fitting with explicit penalty axes,
regularization paths, lasso path screening, and dynamic Gap Safe zero screening
for the three MVP penalties are implemented. Dense Gaussian clustered lasso also
has a hybrid of full proximal-gradient steps and exact signed-cluster coordinate
sweeps; weighted chain fused lasso and SLOPE share its driver with contiguous
segment and signed magnitude-cluster sweeps. This hybrid also fits binomial
models through weighted IRLS subproblems. This document records commitments and
an implementation sequence; illustrative API names are not frozen interfaces.

The next goal is an MVP that the clustered-lasso paper can use and cite through
a public Rust and Python release. The uniform clustered-lasso penalty and dense
Gaussian hybrid from `~/research/clustlasso-cd` are available in Rust.
[TODO.md](TODO.md) tracks hybrid paths, screening, cross-validation, Python API,
and release acceptance checks.

## 1. Purpose

Shrinkage is a composable Rust framework for fitting regularized statistical
models. It must support two equally important uses:

1. A Rust consumer builds one fast, lean solver, such as Gaussian lasso, using
   statically dispatched components and minimal dependencies.
2. R and Python packages expose a broad runtime-configurable collection of model
   families, penalties, solvers, and storage representations.

The ambition is broad composition, including multiple penalties, rather than a
collection of unrelated estimators. This does not promise that every solver
supports every combination. Compatibility and convergence assumptions must be
explicit, with generic methods complementing specialized implementations.

## 2. Mathematical model and scope

The central objective is

$$
  \min_{B,b}\; L(Y, XB + \mathbf{1}b^\mathsf{T} + O; M)
  + \sum_{k=1}^{K}\lambda_k P_k(B),
$$

where `X` is a design operator, `B` is a coefficient vector or matrix, `b` is an
optional unpenalized intercept, `O` contains offsets, and `M` contains metadata
such as weights, event indicators, times, or strata. Constraints may be
expressed as indicator penalties when the solver supports them. The intercept is
excluded from penalties by default; other unpenalized blocks must be explicit.

Here `X` and `B` are on the optimization scale. If preprocessing normalizes the
input matrix, penalties act on the corresponding normalized coefficients unless
an explicitly supported original-scale penalty is requested. Section 8 defines
the initial lasso convention and the transformation of reported coefficients.

Long-term scope:

  | Component | Intended coverage                                                                                                                                                                                    |
  | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Datafits  | Gaussian, multivariate Gaussian, binomial, multinomial, Poisson, other GLMs, Cox and stratified Cox                                                                                                  |
  | Penalties | L1, squared L2/ridge, elastic net, SLOPE, clustered lasso, group and sparse-group penalties, group SLOPE, MCP, SCAD, group variants, other nonconvex penalties, nuclear norm, bounds and constraints |
  | Solvers   | Coordinate and block descent, hybrid clustered-lasso and SLOPE descent, gradient and proximal-gradient methods, FISTA, three-operator splitting and ATOS, ADMM, proximal Newton/IRLS, MM/DC methods  |
  | Storage   | Dense, sparse, memory-mapped, chunked/file-backed, and user-defined operators                                                                                                                        |
  | Utilities | Regularization paths, warm starts, screening, prediction, cross-validation, diagnostics                                                                                                              |

Use `f64` initially. GPU execution is deferred, but the architecture must allow
device-specific implementations. Distributed execution, general symbolic
optimization, out-of-core coefficient vectors, and global optimization of
arbitrary nonconvex objectives are outside the initial scope.

## 3. Architectural boundaries

Keep these responsibilities separate:

- **Matrix capabilities:** shape, products, column operations, and block access.
- **Normalization:** lazy transformations and fitted transformation metadata.
- **Datafit:** loss and derivatives with respect to the linear predictor.
- **Penalty:** value and algorithm-specific capabilities such as a proximal map.
- **Problem:** composition, coefficient layout, intercept policy, and metadata.
- **Solver oracle:** the operations required by a particular algorithm.
- **Solver:** iteration, workspaces, stopping rules, and algorithm diagnostics.
- **Orchestration:** paths, cross-validation, runtime selection, and bindings.

Solvers depend on small capability interfaces. Avoid large traits with optional
methods that fail at runtime. External crates must be able to provide their own
datafits, penalties, matrices, and solvers.

The module layout separates `problem`, `datafit`, `penalty`, `fit`, `model`,
`error`, and `solver`. Named convenience wrappers live in `model/`, individual
datafits in `datafit/`, and fitted result types in `fit/`. Their module roots
define shared capabilities and utilities and re-export the concrete types.
`Lasso` builds a typed `Problem`; `CoordinateDescent` implements
`Solver<Problem<&M, Gaussian, L1>>` and owns the iteration machinery.
`Problem::fit_with` accepts external solvers with their own result and error
types. Component accessors expose the design, datafit, penalty, and
normalization settings to those solvers. Unsupported typed compositions have no
`Solver` implementation for coordinate descent. Selecting coordinate descent for
runtime proximal oracles returns a compatibility error.

`Datafit` and `Penalty` provide fallible value evaluation. Gaussian residual
loss and L1 coordinate minimization are shared with the specialized solver,
whose residual caches and dual certificate remain Gaussian/L1-specific.
`LassoFit` retains the specialized diagnostics and owns original-scale
prediction and back-transformation. `FitError` preserves backend sources;
`LassoError` remains an alias, and existing root, `lasso`, and `problem` import
paths still work. These interfaces are provisional. Runtime proximal oracles and
an experimental block-reader consumer exercise fallible storage access.

`SmoothDatafit` adds a fallible predictor gradient, observation count, and
intercept policy. `ProximalPenalty` requires an exact prox for the complete
convex term. `L1`, `Ridge`, `ElasticNet`, and `Slope` implement this capability.
Ridge means `lambda * ||theta||² / 2`; elastic net accepts separate strengths
for `l1 * ||theta||₁ + l2 * ||theta||² / 2`. There is no inferred prox for a
sum.

`ProximalGradient<V>` uses LazyMatrix's fallible native-vector products and
backtracking against the smooth quadratic upper bound. `ProximalDesign` exposes
complete slice-based forward and transposed products to the shared iteration.
Concrete adapters own normalized LazyMatrix views and native buffers, while
iteration owns slice buffers for scalar updates. `ProximalVector` supplies owned
CPU work buffers for `Vec`, faer `Col`, nalgebra `DVector`, and ndarray
`Array1`. Slice buffers bridge predictor derivatives and complete proximal maps
to those native buffers, with storage reused within a fit and along a penalty
path. This CPU capability does not prescribe future device storage. Shared
problem preparation retains normalization metadata without copying its vectors.
All normalization algebra remains in LazyMatrix, including its scaled
forward-product input clone.

`ProximalFit` owns original-scale parameters, preprocessing, objective, final
step, and stopping diagnostics. The absolute proximal-gradient criterion uses
the mapping's infinity norm and the absolute free-intercept derivative at the
returned iterate. Supported Gaussian and binomial MVP fits additionally report
original-problem certificates when screening is enabled. Unsupported criteria
are rejected, line-search exhaustion returns a distinct error, and iteration
limits remain finite unconverged results. `ProximalGradient::solve_path` and
`solve_runtime_path` prepare the design once and retain native product buffers
and slice storage across penalty points. Each point starts from the preceding
optimization-scale coefficients and intercept, including after an iteration
limit. The step and iteration budget reset at each point. Results own
independent original-scale parameters and preprocessing; result storage still
allocates. Experimental block reading reuses this solver with caller-owned
storage buffers.

`task bench:proximal:heap` measures normalized forward and transposed products,
the elastic-net proximal map, and a path against independent cold fits. It also
checks LazyMatrix 0.5.0's published `matvec_scaled_with_workspace` API with
dense and CSC faer input. The current native adapter uses `MatVecInto`, whose
scaled forward product clones its input. The separate upstream scratch API
avoids that normalization allocation on the measured backends. This does not
establish allocation-free Shrinkage iterations or allocation-free products for
every backend; adopting that API requires its additional fused-product
capabilities.

Build working numerical code before stabilizing abstractions. Do not introduce
an elaborate type system or one crate per prospective feature at bootstrap.

`Irls` adds a typed solver for `DiagonalHessianDatafit` and `CoordinatePenalty`.
Gaussian and binomial losses supply exact diagonal predictor Hessians; L1,
ridge, and elastic net supply exact scalar minimizers and coordinate KKT
violations. External components can implement these capabilities without
providing native matrix products or a complete proximal map. Coupled penalties
do not acquire coordinate updates from a proximal map.

IRLS solves each quadratic model with coordinate descent and an unpenalized
intercept. It maintains gradient residuals rather than dividing by observation
curvatures to construct a working response. Predictor curvature is floored at
`1e-12 / nobs` for the quadratic model only. Binomial curvature uses
`exp(-abs(eta)) / (1 + exp(-abs(eta)))² / nobs` to retain representable tails.
Backtracking halves a full candidate step until the original objective passes an
Armijo check with constant `1e-4`. Decrease of the inner quadratic bounds the
composite directional slope by `-d_eta' H d_eta / 2`, avoiding cancellation
between nearly equal penalty values near convergence.

The default IRLS KKT tolerance is `1e-6`, with 100 accepted outer steps, 1,000
inner sweeps per model, and at most 100 step halvings. Each inner target is
`0.1 * max(current_outer_KKT, outer_tolerance)`. An inner iteration limit can
still yield a valid descent direction. Full original-loss KKT checks include the
intercept, and freshly reconstructed predictors certify convergence and outer
iteration-limit results. A small KKT violation for a separated binomial model
certifies approximate stationarity, not existence of a finite optimum.

`IrlsFit` stores original-scale parameters, preprocessing, objective, full KKT
violation, accepted outer steps, and total coordinate sweeps. Predictions are
linear predictors, including binomial logits. Only absolute KKT stopping is
implemented. The runtime proximal oracles and Python bindings do not expose
IRLS.

## 4. Static computation and runtime composition

The typed Rust API is foundational. Trait bounds express solver requirements,
and static dispatch permits specialized kernels without runtime overhead.

The dynamic facade constructs runtime specifications, validates compatibility,
selects an implementation, and enters numerical kernels. Erase components or
solver-facing oracles at coarse boundaries: a matrix product, proximal
operation, block update, or complete coordinate sweep. Avoid virtual calls per
matrix entry or scalar arithmetic operation.

Do not compile the entire Cartesian product of matrix, datafit, penalty, layout,
and solver types for R/Python. General solvers should accept object-safe oracles
where practical, while important specialized combinations can be explicitly
monomorphized. Keep both paths backed by the same mathematical components.

For the initial CPU proximal solver, `RuntimeProblem` composes independently
boxed `RuntimeDesign`, `SmoothDatafit`, and `ProximalPenalty` objects.
`MatrixDesign<M, V>` adapts a borrowed backend independently of the other
components. Its object-safe preparation resolves normalization using the
datafit's intercept policy, validates concrete columns, and constructs a
`ProximalDesign` oracle with native product buffers. Consuming this oracle moves
the fitted normalization vectors into `ProximalFit` without cloning them.

The same iteration routine accepts concrete and erased design, datafit, and
penalty oracles. Runtime iteration uses `ProximalGradient<Vec<f64>>` options and
concrete slice storage, regardless of backend vector types. Virtual calls occur
at whole-operation boundaries, outside scalar loops. Backend errors erase to
`BackendError`, which retains the original error and its source chain. Erasing a
fully instantiated `Problem<Matrix, Datafit, Penalty>` afterward does not avoid
constructing the Cartesian product. Specialized coordinate solvers may erase a
complete sweep.

LazyMatrix's `Columns` and `RawColumns` use generic associated view types, and
`LogicalColumn` has generic methods. They are static capabilities, not runtime
trait-object interfaces. Keep them inside concrete adapters. The composition
milestone must demonstrate runtime construction from independently selected
operators, datafits, and penalties as well as agreement with typed fits.

An `auto` policy considers mathematical compatibility, storage access, memory,
and available implementations. Initially use a documented deterministic policy;
defer a sophisticated planner. Report the chosen solver and allow overrides.
Unsupported combinations must produce actionable errors.

The initial runtime API requires smooth datafits and complete proximal terms at
construction through trait bounds. `FitError::UnsupportedCombination` identifies
the selected solver, the incompatible selection or missing capability, and a
supported alternative. Unsupported stopping rules return this error in both
typed and runtime fits. `CoordinateDescent` rejects `RuntimeProblem` before
preparation because the proximal oracles lack its Gaussian/L1 coordinate updates
and dual certificate. Its error directs callers to runtime proximal gradient or
a typed Gaussian/L1 problem. External components can return the same error for
their own compatibility requirements, and runtime dispatch preserves it.

## 5. Matrices and LazyMatrix

Use `lazymatrix` as the foundation for matrix operations and just-in-time
normalization. Reuse its existing capabilities and backend adapters, including
dense and sparse integrations, rather than defining a competing matrix API.
Inspect dependency upgrades before changing the pinned release or relying on new
capabilities.

The crate pins the inspected release `lazymatrix = "=0.5.0"`, with default
features disabled. Shrinkage's versioned features `faer_v0_24`,
`nalgebra_v0_34`, `ndarray_v0_17`, and `sprs_v0_11` forward to the matching
LazyMatrix features. The short names `faer`, `nalgebra`, `ndarray`, and `sprs`
remain aliases for these release lines. Gate tests, doctests, examples, and
benchmarks on the versioned features so either entry point works. These retain
faer 0.24, nalgebra 0.34 with nalgebra-sparse 0.11, ndarray 0.17, and sprs 0.11.
ndarray arrays and borrowed views support row-major, column-major, and strided
storage. sprs CSC matrices and views use LazyMatrix's checked `SprsCsc` wrapper;
callers explicitly convert CSR input to CSC before fitting. Version 0.5.0
supplies fallible statistics and products, including a combined
normalization-statistics hook. Keep upstream development overrides local so the
regular fitting APIs build without the sibling repository.

Shrinkage exposes one release line per backend. LazyMatrix 0.5.0 implements
every enabled release independently, retaining older matrix types when feature
unification enables another release.

The `experimental-block-reader` feature prototypes caller-buffered rectangular
access with the published LazyMatrix `ReadBlock` trait.
`experimental::BufferedDesign` borrows one reusable buffer, implements raw
forward and transpose products, and forwards column statistics without changing
their combined scan hook. `experimental::fit_buffered` validates finite entries,
shares normalization preparation, and reuses proximal iteration and fitted
parameter transformation. The `zarrs_v0_22` feature and `zarrs` alias supply its
storage backend. This path keeps decoding out of borrowed-column traits and
discards failed blocks before accumulation. Observation and coefficient vectors
remain in RAM. Strict byte budgets, caching, and reader stabilization remain
future work.

For centering vector `c` and diagonal scale matrix `S`, optimize using

$$
  \widetilde X = (X - \mathbf1c^\mathsf T)S^{-1}.
$$

Apply transformations algebraically:

$$
  \begin{aligned}
    \widetilde Xv            & = X(S^{-1}v) - \mathbf1(c^\mathsf{T}S^{-1}v),    \\
    \widetilde X^\mathsf{T}u & = S^{-1}(X^\mathsf{T}u - c\mathbf1^\mathsf{T}u).
  \end{aligned}
$$

Sparse centering must never materialize a dense matrix. Coordinate methods must
use logical normalized column operations and algebraic centering corrections,
without allocating dense normalized columns. Preserve efficient sparse updates
where the algorithm permits them.

The initial sparse Gaussian solver owns a residual `r = r_base + a * 1`, its
cached base sum, and column summaries. A coordinate update modifies only the
stored column entries and the scalar offset, taking `O(nnz_j)` time without an
`O(n)` broadcast. A cached-sum column dot has the same bound. Initialization,
residual refreshes, and full convergence checks are separate passes. Test these
operation counts and compare reconstructed residuals against dense references;
periodically refresh cached state to control floating-point drift.

Reusable output does not imply allocation-free execution. In release 0.5.0,
`LazyMatrix::matvec_into` clones its input when scaling is active. Measure that
allocation in the proximal-gradient consumer. Before claiming an allocation-free
normalized iteration, add and verify reusable normalization scratch storage in
LazyMatrix. Solvers own or retain their workspaces across iterations and path
points; backend products and proximal operations must report any remaining
allocations. Do not duplicate normalization kernels in Shrinkage to bypass this
boundary.

  | Algorithm                              | Principal design access                                        |
  | -------------------------------------- | -------------------------------------------------------------- |
  | Gradient/proximal gradient, FISTA, TOS | Forward and transposed products                                |
  | Coordinate descent                     | Efficient logical column operations                            |
  | Group/block descent                    | Efficient column-block operations                              |
  | Streaming gradients                    | Sequential row blocks and blockwise datafit evaluation         |
  | Newton/IRLS                            | Weighted products or Hessian-vector operations                 |
  | ADMM                                   | Operators and linear solves determined by the chosen splitting |

Cox risk-set operations belong to datafit/solver logic, not a general matrix
trait. Blockwise loss evaluation is not automatically valid for datafits that
couple observations, such as Cox partial likelihood.

Prefer extending LazyMatrix for generally useful missing capabilities, including
in-place products or fallible block access. Keep solver-specific composite
operations in Shrinkage. Avoid scalar-indexing APIs as the universal interface.
File-backed failures must propagate as errors rather than panic or corrupt a
fit.

Products, column statistics, and computed normalization return `Result` in
LazyMatrix 0.5.0. The lasso uses the combined `normalization_stats` hook and
preserves preprocessing errors with their operation context and source type.
Borrowed column views remain infallible and must not hide I/O or decoding. The
separate `ReadBlock` capability binds each successful block to caller-owned
storage. The experimental consumer reuses a bounded buffer and tests injected
read failures during preprocessing, backtracking, and later solver iterations.
Failed blocks never reach accumulation or loss evaluation. Broader file-format
support remains a later milestone.

## 6. Datafits, penalties, and solver oracles

A datafit owns or borrows response data and metadata. Its basic operations are
loss evaluation and, when supported, gradient with respect to the predictor.
Curvature, coordinate updates, streaming evaluation, and risk-set operations are
additional capabilities, not mandatory methods on every datafit.

A penalty exposes its value. Separate capabilities describe proximal evaluation,
coordinate/block subproblems, convexity assumptions, and DC decompositions.
Distinguish squared L2 (ridge) from the unsquared L2 norm.

Composition should allow conceptual types such as `Scaled<P>`, `Sum<P, Q>`,
`OnRows<P>`, `OnGroups<P>`, and explicit unpenalized masks. Validate group
definitions and distinguish disjoint from overlapping groups.

`Scaled<P>` multiplies a complete penalty value by a finite, nonnegative scalar,
checked when evaluated or fitted. It provides a complete proximal map only when
the inner penalty does: positive scaling replaces the step by `step * scale`,
and zero scaling returns zero value and the identity map. Invalid steps,
dimensions, coefficients, or inner parameters remain errors at zero scaling.
`Penalty::validate_input` defaults to value evaluation; built-in penalties and
wrappers override it to avoid unused value arithmetic at zero scaling. Positive
scaling propagates inner arithmetic errors and rejects nonfinite scaled values,
overflowing steps, or positive steps that underflow to zero. The wrapper owns no
workspace and introduces no allocations.

`Masked<P>` owns an explicit boolean mask with one entry per optimization-scale
coefficient, excluding intercepts. A true entry makes that coordinate
unpenalized. The restricted value equals the inner penalty with excluded
coordinates set to zero, and the exact restricted prox copies those inputs
unchanged. `RestrictedProximalPenalty` supplies these operations and a separate
validation hook. L1, ridge, and elastic net implement it; `Scaled` and `Box`
preserve it. Masking a penalty without this capability is a type error,
including coupled SLOPE and an erased `Box<dyn ProximalPenalty>`. A coupled
penalty needs its own restricted implementation; applying its complete prox and
restoring excluded inputs does not in general solve the restricted problem.
Length, coefficient, and parameter validation applies even to fully excluded
masks or zero outer scales. Built-in restrictions skip excluded value arithmetic
and allocate no workspace during evaluation.

`DisjointGroups` owns coefficient index vectors and one explicit finite,
nonnegative weight per group. Construction rejects empty groups, repeated
indices, duplicate or overlapping groups, out-of-range indices, and mismatched
weight counts. Group and member order are preserved, and weights have no
implicit group-size or group-count scaling. Indices address optimization-scale
coefficients, excluding intercepts; flattened matrices use `CoefficientLayout`'s
response-major order. `UncoveredPolicy::Reject` requires full coverage, while
`UncoveredPolicy::Unpenalized` records omitted indices in increasing order for
unpenalized handling. Zero-weight groups still count as coverage and undergo all
membership checks. An empty collection is valid for zero coefficients under
either policy, or for any coefficient count with unpenalized handling.

`GroupLasso::new(lambda, groups)` implements the complete convex penalty
`lambda * sum_g weights[g] * ||theta[g]||₂`. Its exact prox multiplies each
block by `max(1 - step * lambda * weights[g] / ||input[g]||₂, 0)` and maps
zero-norm blocks to zero. Zero-strength and zero-weight blocks, along with
uncovered coordinates, retain their inputs. Singleton groups give weighted L1.
Strengths must be finite and nonnegative, and input dimensions must match the
validated definitions even at zero strength. Finite inputs, positive finite
steps, and finite penalty and threshold arithmetic are checked independently.
Scaled norms avoid squaring coefficients on their original scale and support
proximal inputs whose block norm exceeds `f64::MAX`. Neither value evaluation
nor the prox allocates scratch storage.

Typed and runtime Gaussian problems use the existing proximal-gradient
iteration, native product buffers, and reusable slice workspaces. The objective
is `||y - X theta - b||² / (2n) + lambda * sum_g weights[g] * ||theta[g]||₂`,
where `X` and `theta` denote the optimization scale. The intercept remains
unpenalized. Fitted normalization and back-transformation follow the same
prediction-preserving contract as other penalties; group norms act on
optimization-scale coefficients rather than reported original-scale values. This
concrete penalty does not introduce generic group composition or automatic path
thresholds.

Tests certify the block subgradient equations, zero blocks, singleton L1
equivalence, zero weights, and free coordinates. A correlated Gaussian problem
has a constructed KKT solution with active and inactive blocks and an uncovered
coefficient, providing independent coefficients, residual correlations, and an
objective reference. Explicit and lazy normalization agree with and without a
free intercept. Native dense, strided, and CSC adapters exercise typed and
runtime fitting. Run `cargo run --example group_lasso --features faer_v0_24` for
an orthogonal Gaussian example, `task bench:groups` for block operations and
complete dense and CSC fits, and `task bench:proximal:heap` for allocation
checks.

The proximal map of a sum is not generally the composition of its individual
proximal maps. Consequently:

- Proximal gradient requires a valid prox for the complete nonsmooth term.
- Three-operator splitting can handle two separately proximable terms under its
  stated assumptions.
- ADMM requires an explicit splitting with supported subproblem solvers.
- Specialized formulas may handle compositions such as elastic net or particular
  sparse-group structures.

Solver-facing oracles expose only the operations an algorithm needs. Examples
include smooth value/gradient plus prox, a three-operator oracle, a coordinate
sweep oracle, or an ADMM split oracle. CPU implementations may use host slices;
these are not universal storage requirements for all backends.

Solver-facing operations that can encounter backend failures return `Result`. An
adapter over an in-memory LazyMatrix operation returns `Ok` after validating
dimensions at the fit boundary. Fallible adapters preserve the source error and
operation context through preprocessing, products, and fitting. On failure,
partially written output buffers are invalid, and the fit returns an error; it
must not report optimization success or consume that output in another step.
This error channel is separate from ordinary optimization termination.

Convex and nonconvex algorithms must have distinct documented assumptions.
Nonconvex convergence reports must identify a stationarity criterion when one is
available; a small objective change alone is not proof of stationarity or a
global optimum. Nonconvex smooth TOS results must not be assumed to cover
arbitrary nonconvex nonsmooth penalties.

### Uniform clustered lasso and the Gaussian hybrid

`ClusteredLasso::new(l1, fusion)` defines the complete convex penalty

$$
  R(\theta) = \lambda_1\sum_j|\theta_j|
              + \lambda_2\sum_{i<j}|\theta_i - \theta_j|.
$$

Both strengths are finite and nonnegative. Each unordered pair contributes once,
regardless of the signs of its coefficients. Empty vectors are valid. Intercepts
are excluded. The fitted Gaussian objective is
`||y - b - X_tilde theta||² / (2n) + R(theta)`. Strengths and cluster labels
refer to optimization-scale coefficients. Back-transformation preserves
predictions, but unequal column scales can turn a fused optimization block into
unequal original-scale coefficients.

The reference is research revision `293760ea4040351a654906add5d7ca855be1deb9`.
The source contracts are `docs/cluster-updates.md`, `docs/pg-partition.md`,
`docs/convergence.md`, and the backtracking addendum `docs/line-search.md`.
Their SHA-256 hashes are recorded in `tests/clustered_lasso/provenance.json`.
The contracts below and committed fixtures are self-contained; builds and tests
never read that checkout.

**Scaling the research objective.** With `Normalization::None` and no fitted
intercept, divide both research strengths by `n`. Then
`F_shrinkage = F_research / n` at every coefficient vector, and scalar
minimizers are unchanged. Divide research curvatures and scalar linear terms by
`n` as well. Corresponding PG curvatures obey `L_shrinkage = L_research / n`,
giving identical proximal inputs and outputs and `G_shrinkage = G_research / n`.
If research stopping uses `||G_research||_inf / s <= tolerance`, where
`s = max(1, ||X' y||_inf)`, use Shrinkage's absolute mapping tolerance
`tolerance * s / n`. Absolute objective differences scale by `1/n` too. A zero
design admits any positive curvature; choose corresponding positive values
explicitly when comparing maps rather than assuming identical fallback values.

**Exact complete prox.** Sort signed input values in ascending order, with
feature indices breaking ties. For zero-based rank `i`, shift the sorted value
by `step * fusion * (p - 1 - 2i)`. Project these values onto nondecreasing
sequences through PAVA, then soft-threshold each pooled level by `step * l1` and
restore feature order. This composition is the exact prox for this specific
complete penalty. Evaluate the fusion value through adjacent sorted gaps: gap
`i` has multiplicity `i * (p - i)`. Skip zero-strength terms and zero gaps.
Value and prox reuse `O(p)` scratch and take `O(p log p)` time. Positive scaled
strengths that underflow to zero and nonfinite derived arithmetic return errors.

**Signed state and exact sweeps.** Canonical blocks contain exactly equal signed
values, sorted increasingly, with feature indices sorted inside a block.
Normalize both signed zeros to positive zero. Opposite signs, merely close
values, and adjacent unequal floats remain distinct. Each block stores its
membership, aggregate `a_C = sum_{j in C} X_tilde[:,j]`, averaged curvature
`q_C = ||a_C||²/n`, and a stable identity. The maintained residual is
`r = y - b - X_tilde theta`.

For a block of size `m` and value `c`, the exact scalar restriction is

$$
  \phi(z) = q_Cz^2/2 - h_Cz + \lambda_1m|z| + \lambda_2m\sum_{D \ne C}|D||z
            - c_D|, \qquad h_C = a_C^\top(r + a_Cc)/n.
$$

Combine coincident knots and discard zero weights. For positive curvature, scan
ordered knots using the interval root `(h_C + W - 2P)/q_C`, where `W` is total
knot weight and `P` is weight strictly to the left. At a knot, its left and
right derivatives bracket zero exactly when that knot minimizes the restriction.
For exactly zero curvature, the aggregate column and linear term must be zero;
project `c` onto the weighted-median interval, retaining `c` when there are no
positive weights. Never round a small positive curvature to zero. Nonzero
aggregates with underflowed zero curvature produce a numerical error.

A sweep snapshots block identities in initial signed-value order. Each surviving
identity is visited once. Moves may cross any number of blocks and may move a
zero block. Update `r -= a_C * (z-c)`. Merge exact ties immediately, retain the
moving identity, and retire absorbed identities so they cannot be visited later
in that sweep. Add aggregate columns and recompute curvature, including cross
terms. Minimize an enabled free intercept exactly after each sweep. Every exact
scalar minimization is nonincreasing in the original complete objective.

**PG transitions, bounds, and convergence.** `Hybrid<V>` starts with a full PG
transition, then performs five complete sweeps between PG transitions by
default. Every transition directly refreshes the residual before its full
gradient and after its candidate. Rebuild membership and all aggregate columns
from the candidate in original feature order. Stage this state before committing
coefficients, intercept, and curvature together. The rebuild allows zero and
nonzero blocks to split, even when block count is unchanged.

`HybridStep::FixedBound` uses `||A||_F²/n` with a conservative rounding guard,
where `A` augments the normalized design with a column of ones if an intercept
is fitted. Use one for a zero bound. `HybridStep::Backtracking` defaults to the
largest averaged column curvature, including intercept curvature one, and uses
one when all curvatures vanish. A supplied positive curvature is a seed. Double
it until `||A d||²/n <= L ||d||²` passes, then retain the accepted value. Scale
nonzero displacements before applying the design, compare Euclidean norms with
relative slack `64 * f64::EPSILON`, and avoid subtraction of nearly equal
losses. Exhausted searches return `LineSearchFailure`. The fixed bound is
mathematical; neither policy claims a floating-point proof of exact descent.

In exact arithmetic, a full PG step with an adequate bound decreases the
objective by at least `L * ||d||²/2`. Finite nonincreasing cluster sweeps
between infinitely many such PG steps preserve objective convergence to the
global minimum. Backtracking retains positive curvatures bounded above by the
larger of its initial value and twice the augmented spectral curvature. These
bounds give the same descent and limiting-stationarity argument. Rank deficiency
is allowed: compactness is needed only modulo directions annihilated by both the
design and penalty. Fitted values converge to the common optimal fit;
coefficient convergence requires uniqueness or additional assumptions. This
argument does not promise finite identification, a runtime advantage, or a rate.

**Stopping, storage, and interfaces.** Convergence requires the infinity norm of
the full PG mapping, maximized with the absolute free-intercept derivative, at
the returned coefficients. Its default absolute tolerance is `1e-6`. Check
initially, every ten iterations by default, and at the iteration budget. One
iteration is one committed PG transition or complete sweep; the default budget
is 10,000. Checks use freshly computed residuals in separate scratch and neither
commit diagnostic candidates nor change the schedule, maintained CD residual, or
accepted curvature. Unchanged sweeps never establish convergence.

`HybridFit` reports the final objective, mapping and threshold, step policy,
step size, termination reason, cluster labels and values, PG steps, coordinate
sweeps, diagnostic checks, and backtracks. Finite budget exhaustion reports
`IterationLimit`. Screening-enabled fits report an original-problem duality gap
independently of the stopping criterion. Typed Gaussian problems use the dense
hybrid; runtime proximal oracles support the penalty through PG and reject
hybrid selection before preparation. Hybrid paths, fusion screening, and
sparse-specific extensions remain later milestones.

Dense aggregate storage is `O(nK + p + n)` per state. Staging retains both
committed and candidate aggregates, with buffers kept at their high-water
cluster count. At `K=p`, each aggregate set costs approximately `8np` bytes,
comparable to an additional dense design. Membership buffers are reused and
retired large memberships are released to keep their storage linear in `p`. An
ordered-knot sweep costs `O(nK + K²)` plus up to `O(pK)` for maintaining feature
labels and merged memberships. Full PG transitions cost `O(np + p log p)`
including complete rebuilds. Native product adapters retain their existing
normalization allocation behavior.

Independent subgradient certificates and exhaustive ordered partitions verify
the prox. Committed explicit-pair CVXPY/Clarabel fixtures include rank-deficient
designs, zero-strength limits, and free intercepts; their duals independently
certify KKT conditions. Tests compare complete sweeps with rebuilding every
scalar restriction, and cover required splits, signed zeros, crossings, merges,
diagnostic independence, budgets, normalization, and native dense backends.
`task bench:clustered` measures penalty operations and complete solves at
matched accuracy on clustered and weakly fused data. `task bench:clustered:heap`
checks warmed scratch reuse and measures both fused and nearly singleton
aggregate storage. No timing or allocation claim excludes setup, rebuilds, or
final checks unless explicitly labeled.

### Weighted chain fused lasso and segment sweeps

`ChainFusedLasso::new(l1, fusion)` defines

$$
  R(\theta) = \lambda_1\sum_j|\theta_j|
              + \lambda_2\sum_{j=0}^{p-2}w_j|\theta_{j+1} - \theta_j|.
$$

Strengths are independent, finite, and nonnegative. Weights default to one;
`.with_weights(Vec<f64>)` supplies exactly `p.saturating_sub(1)` finite,
nonnegative weights. Validate dimensions and weights even at zero strength.
Empty and single-feature chains have no edges. Each edge contributes once, in
training-column order, without dividing by the edge count. The Gaussian
objective is `||y - b - X_tilde theta||² / (2n) + R(theta)`. Intercepts are
unpenalized. Penalties and segment labels use optimization-scale coefficients;
back-transformation preserves predictions, but unequal column scales may change
equalities among reported coefficients.

**Complete exact prox.** Split at zero-weight edges and solve weighted total
variation on each subchain with tube half-widths `step * fusion * weights[j]`.
Use the classic taut-string method with reusable upper and lower hull deques,
then soft-threshold by `step * l1`. The weighted construction follows Sections
2.1 and 2.4 of [Barbero and Sra (2018)](https://arxiv.org/html/1411.0589v3).
Every hull piece enters and leaves a deque once, giving `O(p)` time and `O(p)`
scratch without an iterative tolerance. Compensated heights preserve small
signal increments with large tube widths. Warmed calls at a fixed dimension and
input allocate no new storage.

To justify the composition, let `u` be the TV minimizer, with edge dual `q`, and
let `z` be its uniform soft-thresholding. Nonzero differences in `z` retain
their signs from `u`; differences collapsed to zero accept the same bounded
dual. Thus `q` remains a TV subgradient at `z`, while `(u - z) / step` is an L1
subgradient at `z`. Their sum certifies the complete proximal stationarity
equation, including nonuniform and zero weights. This argument does not assert
arbitrary composition of proximal maps. Reject nonfinite derived arithmetic and
positive thresholds that underflow to zero; skip inactive terms.

**Segment state and sweeps.** Segments contain contiguous, exactly equal signed
coefficients. Normalize signed zeros, and never merge merely close values.
Zero-weight edges always separate segments, including at zero fusion strength.
Equal values on nonadjacent segments remain separate. Store feature ranges,
stable identities, aggregate columns, averaged curvatures, boundary weights, and
residuals. For segment `C`, with neighboring levels `c_L` and `c_R`, minimize

$$
  q_C z^2/2 - h_C z + \lambda_1|C||z| + \lambda_2 w_L|z - c_L|
  + \lambda_2 w_R|z - c_R|, \qquad h_C = a_C^\top(r + a_C c_C)/n.
$$

Missing boundaries contribute nothing. Sort and combine the at most three knots,
then use the shared exact scalar minimizer and its zero-curvature
weighted-median rule. A sweep snapshots identities from left to right and visits
each surviving identity once. Moves can cross neighboring levels without
changing feature order. Merge adjacent exact ties across positive weights
immediately, retain the moving identity, and retire absorbed identities. Add
aggregate columns and recompute curvature, including cross terms. Update
residuals for each move, and minimize an enabled intercept after the sweep.

**Hybrid integration.** Typed Gaussian problems use `Hybrid<V>` with the same
schedule, line search, stopping tolerance, and budget semantics as clustered
lasso. Private partition adapters share the driver and preserve penalty-specific
membership. Rebuild all segments and aggregates after accepted full PG steps so
segments can split. A fresh full-space PG mapping, including the free-intercept
derivative, certifies convergence at the returned point. Unchanged sweeps do
not. The clustered-lasso descent and convergence argument applies because
segment sweeps exactly minimize restrictions of this convex objective.

`HybridFit::cluster_labels()` indexes segment values in feature order for this
penalty. `cluster_values()` may repeat and is not sorted. Clustered lasso
retains its strictly increasing value order. PG supports typed and runtime
problems; runtime hybrid selection remains unsupported. Screening-enabled fits
report original-problem dual certificates and gaps.

Each dense aggregate set uses `O(nK + p + n)` storage for `K` segments, with
committed and staged sets retaining high-water capacities. Full PG transitions
and rebuilds take `O(np + p)` on dense designs. Scalar restrictions use only
boundary knots; merges can still require `O(pK + K²)` label and range
bookkeeping over a sweep, in addition to `O(nK)` aggregate work.

Independent epigraph-QP fixtures and their dual certificates verify proximal
maps and Gaussian fits. Sweep tests rebuild each restriction independently and
check reconstruction, residuals, curvature cross terms, objective descent, cuts,
crossings, merges, and PG-required splits. The `chain_fused_lasso` example
includes nonuniform weights, a cut, normalization, and a free intercept.
`task bench:chain` compares proximal operations and complete solves at matched
accuracy; bounded runs include setup, first PG, and subsequent rebuild work.
`task bench:chain:heap` checks warmed scratch reuse and complete-solve storage
for strongly fused and nearly singleton partitions. Hybrid paths, fusion
screening, Python exposure, and sparse-specific optimizations remain later
milestones.

## 7. Specialized SLOPE support

`Slope` implements the generic sorted L1 penalty with an exact proximal map. It
accepts one finite, nonnegative, nonincreasing weight per coefficient, including
the strength, and validates weights and dimensions when evaluated or fitted.
Magnitudes are sorted on the optimization scale. Pooling adjacent violations
projects the thresholded magnitudes onto nonincreasing sequences; clipping at
zero and restoring signs and order completes the prox. Owned CPU sorting and
block buffers are reused across calls. Block means avoid overflowing sums when
the solution remains representable. Empty coefficients require empty weights,
and zero weights define an unpenalized term.

Independent exhaustive partition solutions and subgradient certificates test
proximal optimality, including ties and zero weights. Typed and runtime Gaussian
fits agree with an orthogonal solution. `task bench:slope` measures distinct and
tied magnitudes, and `task bench:proximal:heap` checks value and prox
allocations.

**Gaussian hybrid and state.** Typed Gaussian problems also support `Hybrid<V>`.
Its driver, controls, line search, full-space stopping checks, and diagnostic
accounting are shared with the fusion penalties. SLOPE has a private partition
implementation following [Coordinate Descent for
SLOPE](https://proceedings.mlr.press/v206/larsson23a.html). The complete
objective is
`||y - b - X_tilde theta||² / (2n) + sum_j weights[j] * |theta|_(j)`, with an
unpenalized intercept. No additional strength or observation-count factor
multiplies the supplied weights.

Canonical blocks group exactly equal magnitudes in strictly decreasing order,
with feature indices sorted inside each block. Opposite signs can share a block;
adjacent unequal floating-point magnitudes remain distinct. Both signed zeros
belong to one zero block, with magnitude and individual signs zero. For each
nonzero block `C`, retain signs `s_j` and the signed aggregate
`a_C = sum_{j in C} s_j X_tilde[:, j]`, together with its averaged curvature
`q_C = ||a_C||²/n`. Prediction is `b + sum_C a_C c_C`. The zero block has a zero
aggregate and stays fixed throughout restricted sweeps.

**Exact scalar restriction.** Updating a nonzero block of size `m` replaces
`theta_j` by `s_j z` for all its members, while other coefficients stay fixed.
With current magnitude `c`, maintained residual `r`, and
`h = a_C' (r + a_C c)/n`, minimize

$$
  \tfrac12 q_C z^2 - h z + R(\theta(z)).
$$

The penalty restriction is even in `z`. For positive curvature, solve for
`u = |z| >= 0` using linear target `|h|`, then restore the sign of `h`. Within
an interval between other positive block magnitudes, let `r_u` count other
features with magnitude above `u`. The penalty derivative is
`W(r_u) = sum(weights[r_u..r_u+m])`. At zero, return zero if `|h| <= W(r_0)`.
Otherwise scan increasing positive magnitudes, using `( |h| - W ) / q_C` inside
each interval and the one-sided derivatives at each knot. A derivative jump
containing zero selects the exact knot, allowing a merge. Crossings may pass any
number of blocks; the minimizer is not restricted to adjacent ranks.

Compute the initial rank-window sum with compensated summation. When crossing a
block of size `d`, update it through the nonnegative increments
`sum_{j=r_u-d}^{r_u-1} (weights[j] - weights[j+m])`. This avoids cancellation
from subtracting large prefix sums when trailing weights are small. Repeated and
zero weights remain valid. With exactly zero aggregate curvature, the linear
term must be zero. Project the current magnitude onto the penalty's minimizing
interval: zero if the initial derivative is positive, otherwise the interval
from zero to the first knot with a positive derivative jump, or the whole
nonnegative line when every derivative vanishes. Never round a small positive
curvature to zero; nonzero aggregates with underflowed curvature and nonfinite
derived arithmetic return numerical errors.

**Sweeps and PG transitions.** Snapshot nonzero block identities in descending
magnitude order, visiting each surviving identity once. Update residuals with
`r -= a_C (z-c)` before changing the block's orientation. A negative scalar
flips every member sign and negates its aggregate. A zero scalar clears its
signs and aggregate. Merge exact magnitude ties immediately, preserving the
moving identity and retiring the absorbed identity. Sum the oriented aggregate
columns and recompute curvature with all cross terms. Minimize the enabled
intercept after each sweep. These are exact restrictions of the complete convex
objective, so the common hybrid descent and convergence argument applies,
including repeated and zero weights.

Begin with a full PG transition, then perform five sweeps per PG by default.
Rebuild ordering, signs, memberships, and aggregates after every accepted PG
transition. PG can split nonzero blocks and reactivate individual zero
coordinates, including after unchanged restricted sweeps. Only a fresh
full-space PG mapping, including the free-intercept derivative, establishes
convergence. Checks do not alter the maintained partition, residuals, curvature,
or schedule. `HybridFit` retains the common budget and diagnostic semantics.
Runtime hybrid dispatch remains unsupported; runtime PG remains available.

**Normalization and results.** Lazy normalization and supplied preprocessing use
the same prepared design as PG. Back-transformation preserves predictions.
SLOPE's `HybridFit::cluster_values()` contains nonnegative optimization-scale
magnitudes in strictly decreasing order, with zero last when present.
`cluster_labels()` indexes those values; `coefficient_signs()` contains
optimization-scale signs in feature order, recorded before back-transformation.
Reconstruct `theta_j` as `signs[j] * values[labels[j]]`. Clustered lasso and
chain fused lasso continue to return signed block values. Unequal feature scales
can change equalities and ordering among original-scale coefficient magnitudes.

**Storage and validation.** Each partition retains `O(nK_max + p + n)` storage,
where `K_max` is its high-water block count. Staging a PG rebuild retains both
committed and candidate states. At `K_max=p`, each aggregate set uses roughly
`8np` bytes. Membership and sorting buffers are reused; retired large membership
buffers are released. A scalar restriction scans at most `O(p)` weights and
`O(K)` knots. A sweep spends `O(nK)` on residual updates and `O(pK)` on scalar
scans, with additional membership sorting, label maintenance, and curvature
recomputation when blocks merge. Full PG rebuilds take `O(np + p log p)` time.

Tests compare scalar updates with independent signed-interval enumeration and
complete sweeps with rebuilding each restricted problem. Committed top-k
magnitude epigraph-QP fixtures and dual prefix/support certificates verify
complete Gaussian fits, including rank deficiency, zero designs, intercepts, and
zero-weight limits. Backend tests check normalization, strided views, and
prediction-preserving back-transformation. `task bench:slope:hybrid` compares
complete PG and hybrid solves on constructed distinct, tied, and highly
clustered optima at matched accuracy. Bounded runs include setup, the first PG,
and a subsequent rebuild. `task bench:slope:heap` measures warmed proximal
allocations and complete-solve peak storage at low, high, and changing block
counts. Hybrid paths, fusion screening, Python exposure, and sparse-specific
optimization remain later milestones.

### Logistic hybrid IRLS

Typed `Binomial` problems support the same `Hybrid<V>` and `HybridFit` as
Gaussian problems for all three coupled penalties. The original objective is the
averaged binomial negative log likelihood plus the complete penalty on
optimization-scale coefficients. The intercept remains unpenalized, and
predictions remain logits. Preprocessing is fitted once and reused throughout.

At each accepted outer point, the binomial gradient `g` and diagonal predictor
Hessian `H` define the quadratic model
`g' d_eta + d_eta' H d_eta / 2 + R(theta + d_theta)`. Stable binomial tails and
the model-only floor `1e-12 / nobs` match coordinate IRLS. Internally, weights
are `w_i = nobs * H_i` and residuals are `r_i = -nobs * (g_i + H_i * d_eta_i)`.
This representation never divides by a row weight. For an aggregate `a` at level
`c`, its scalar curvature is `sum(w_i * a_i²) / nobs`, and its linear term is
`sum(a_i * (r_i + w_i * a_i * c)) / nobs`. A move by `delta` subtracts
`w_i * a_i * delta` from each residual. The free-intercept update is
`sum(r_i) / sum(w_i)`. Zero-curvature restrictions retain their penalty-only
minimizers; an all-zero weighted intercept with zero gradient remains fixed.

The private quadratic datafit also supports proximal gradient. Hybrid inner
solves reuse summaries, residuals, and dense aggregate buffers and start with a
full PG transition. Every weight change refreshes curvatures, weighted column
summaries, and the PG curvature policy. Merges recompute curvature from the
combined aggregate, including cross terms. Gaussian kernels retain their
unweighted arithmetic. Storage remains `O(nK + p + n)`, with committed and
staged aggregates retaining their high-water capacities.

The original-loss diagnostic step is fixed at the reciprocal of a guarded
Frobenius bound for `X_aug' X_aug / (4n)`, including the intercept column when
enabled. Both outer and inner mappings use this step, so inner accuracy has a
consistent scale. Each inner target is `0.1 * max(outer_mapping, tolerance)`. An
inner iteration limit can supply a direction, but that direction must decrease
the complete quadratic model. Original-objective Armijo backtracking uses the
coordinate IRLS convention: constant `1e-4`, step halvings, and the directional
upper bound `-d_eta' H d_eta / 2`. Numerical and line-search failures are
errors. Damped steps can change ties, signs, and ordering, so the next inner
solve and final diagnostics rebuild partitions from the accepted coefficients.

`max_iterations` retains its 10,000-step default and counts PG transitions plus
complete sweeps across all subproblems. Logistic controls add 100 accepted outer
steps, 1,000 inner iterations per model, and 100 outer step halvings by default.
`HybridFit::outer_iterations()` and `outer_backtracks()` report outer work; the
existing iteration, PG, sweep, curvature-backtrack, and diagnostic counters
report inner work. Gaussian fits report zero outer work. Original-loss checks
run initially, after each accepted outer step, and before every returned limit
result. Only their full mapping, including the intercept derivative, certifies
convergence. For separated data, a finite tolerance certifies approximate
stationarity rather than existence of a finite exact optimum.

Independent weighted row-scaling references validate the quadratic models,
including zero weights and free intercepts. Logistic tests compare all three
penalties with original-loss PG and independently recompute final objectives and
mappings. They also cover damping, limits, normalization, and separated data.
`task bench:hybrid:irls` measures complete solves and bounded setup/rebuild
work; `task bench:hybrid:irls:heap` checks allocation and peak-storage behavior
across repeated quadratic models. Logistic hybrid paths remain a later step.

## 8. Statistical semantics

Coefficient layouts distinguish features, responses, classes, and groups even
when internal storage is flattened. Penalty axes must be explicit. Multinomial
identifiability conventions must be chosen and documented because
reference-class and symmetric representations change penalty semantics.

The provisional CPU `CoefficientLayout` records the feature-by-response shape
and uses response-major storage: `response * nfeatures + feature`.
`MultivariateGaussian` accepts response-major observations with explicit
observation and response counts. Its loss is `||Y - predictor||_F² / (2n)`,
summing over responses and averaging over observations. `AxisPenalty` applies a
complete vector penalty across `Features` separately for each response, across
`Responses` separately for each feature, or to `All` coefficients at once.
Disjoint group values are summed, each with the supplied strength; response
count never rescales the loss or penalty. Exact vector proximal maps therefore
give exact maps for these disjoint groups, including external terms that couple
responses. A reused scratch buffer gathers strided response groups.

Row composition uses these existing axes with `Scaled` and `Masked` as the inner
vector penalty. On `Responses`, a mask has one entry per response and applies
within every feature row. On `Features`, a mask has one entry per feature and
excludes whole feature rows across responses for penalties with an implemented
restricted proximal map. Both wrapper orders preserve the complete penalty and
proximal map. Each group's strength includes the scale without an additional
feature or response divisor. Masks never include intercepts. Empty layouts and
zero scales still validate the inner parameters and the mask against the
selected vector width.

`ProximalGradient` reuses its existing joint iteration with a stacked design
adapter. The adapter repeats normalized LazyMatrix products for each response
and appends free intercept coordinates with an identity proximal map. It never
materializes a Kronecker design, normalized matrix, or Gram matrix. The internal
stacking leaves the loss's `1/n` convention intact. Intercepts remain separate
from the public coefficient layout and penalty. `MultivariateFit`
back-transforms each response using shared feature preprocessing, retains the
joint objective and mapping diagnostics, and predicts response-major columns.
`RuntimeMultivariateProblem` uses the same adapters and iteration with a
runtime-selected design and vector penalty. These interfaces do not yet specify
class identifiability, overlapping groups, or device storage.

Normalization is a modeling choice. Record the centering/scaling rule and the
scale on which penalties are defined. With normalized coefficients `Theta`,
return original-scale coefficients using

$$
  B = S^{-1}\Theta,\qquad b = \widetilde b - B^\mathsf Tc.
$$

Retain fitted preprocessing metadata and verify prediction equivalence. With no
fitted intercept, centering can induce a fixed original-scale intercept; do not
silently discard it or claim the same no-intercept model was fitted.

The lasso convenience API accepts `.normalize(Normalization)`, replacing the
former `standardize(bool)` option. The enum provides `Auto`, `None`, `Center`,
`Standardize`, `MinMax`, `MaxAbs`, `L1`, `L2`, and `Custom { center, scale }`.
Custom choices reuse LazyMatrix's `Centering` and `Scaling` enums. Centering and
scaling are independent; norm and maximum-absolute scales are computed after
centering. The implementation delegates statistics and normalization algebra to
LazyMatrix.

`Auto` preserves the initial default: fit an intercept, center by
training-column means, and scale by population standard deviations. Disabling
the intercept under `Auto` disables centering while retaining scaling. `None`
uses the raw design. All explicit presets and custom choices are independent of
the intercept option and builder call order. Explicit centering without a fitted
intercept must retain the induced original-scale intercept. Fitted preprocessing
records the resolved centering and scaling rules, values, and whether vectors
were supplied. User-supplied center and scale vectors independently override the
corresponding rule. Nontranslation-invariant computed scales use supplied
centers.

The initial penalty always acts on the optimization coefficients. Original-scale
penalties under nontrivial scaling are deferred until their transformed
subproblems are implemented explicitly.

Specify loss normalization, observation-weight semantics, offsets, and penalty
scaling. For the initial Gaussian lasso, use

$$
  \frac{1}{2n}\|y - \widetilde X\theta
  - \widetilde b\mathbf1\|_2^2 + \lambda\|\theta\|_1.
$$

Without normalization, `X_tilde = X` and `theta = beta`. With scaling, an
original-scale lasso penalty would instead be `lambda * ||S^-1 theta||_1`;
back-transforming fitted coefficients does not change the penalty that was
optimized. The same distinction affects SLOPE ordering. Objectives, gradients,
KKT tolerances, and automatic `lambda_max` calculations use the `1/n` loss
convention. When adapting an unaveraged least-squares coordinate formula, its
threshold is `n * lambda`.

Choose and test policies for constant columns, missing/nonfinite values, and
invalid scales. Do not silently impute data. Fit preprocessing separately on
each cross-validation training fold; reuse it along that fold's entire path.

For the first solver, reject zero observations, nonfinite inputs, negative or
nonfinite penalty strengths, and explicit scales that are not finite and
strictly positive. Replace computed zero scales with one, as LazyMatrix does,
and hold zero-norm normalized columns at zero. Validate inputs before invoking
LazyMatrix constructors: the matrix library deliberately preserves nonfinite
statistics and does not supply the statistical fit's validation policy.

The implemented `Lasso` builder defaults to a relative duality-gap tolerance of
`1e-6` and at most 10,000 complete cyclic sweeps. At zero penalty, the default
uses absolute KKT tolerance `1e-6`, because a residual-scaled feasible dual
vector can give a loose lower bound. For `c_j = X_tilde_j' r / n`, use
`|c_j - lambda sign(theta_j)|` on active coordinates and
`max(|c_j| - lambda, 0)` on zero coordinates, together with `|mean(r)|` when
fitting an intercept. Check the maximum violation after each sweep. Reconstruct
residuals every 50 sweeps, at the iteration limit, and before accepting
convergence. An initially optimal fit takes zero sweeps.

`LassoFit` owns original-scale parameters, fitted centers and scales, and final
objective and KKT diagnostics. A finite fit that exhausts its budget returns
`Termination::IterationLimit`. Invalid inputs, nonfinite arithmetic, and backend
preprocessing errors return distinct `LassoError` variants. The initial typed
`Problem<Design, Gaussian, L1>` pairs these components with `CoordinateDescent`
and uses the same fitting and back-transformation path as `Lasso`. This first
combination does not yet provide generic solver capabilities or runtime
selection; those belong to the composition milestone.

Paths support warm starts, reusable working sets, and full optimality checks.
Automatic maximum-penalty calculations and sequences are
family/penalty-specific. For multiple penalties, allow independent strengths and
meaningful mixing parameterizations without assuming every tuning problem is
one-dimensional.

`PathSolver` provides owned explicit penalty paths through
`Problem::fit_path_with`. `RegularizationPathSolver` adds maximum-penalty
queries and `fit_regularization_path_with`, generating the sequence within one
prepared fit. `Lasso` supplies corresponding convenience methods. `PathFit`
retains ordered fitted models, complete penalty settings, and automatic sequence
metadata. Existing proximal path methods retain their vector return types.
Explicit points preserve caller order, including repeated, increasing, and mixed
penalties. All requested points are fitted, iteration-limit points warm the next
fit, and errors return no partial path. Results own independent original-scale
parameters and preprocessing.

`NullModelDatafit` and `PathPenalty` are opt-in capabilities for exact null
intercepts, penalty thresholds, and strength construction. `PathDatafit`
combines the null-model and smooth-loss capabilities for runtime erasure.
`RuntimePathProblem` and `ErasedPathPenalty` retain automatic capabilities
before boxing, while ordinary runtime explicit paths continue to accept
arbitrary smooth datafits and complete proximal terms. Dispatch stays outside
scalar loops.

Automatic sequences use 100 log-spaced points by default. Their endpoint ratio
is `0.01` when observations are fewer than features, or `0.0001` otherwise.
`PathOptions` overrides length, ratio, or starting strength. A zero start yields
the requested number of zero values. Thresholds use the optimization-scale
gradient at zero coefficients with enabled intercepts optimized. Gaussian null
intercepts are response means, including one per multivariate response. Binomial
uses the log odds of the response totals and rejects boundary responses for
automatic free-intercept paths because their null optimum is not finite.

L1 uses the null gradient's infinity norm. Elastic net follows glmnet's mixing
parameterization: `l1 = lambda * alpha` and `l2 = lambda * (1 - alpha)`.
`ElasticNet::from_alpha` retains alpha even at zero strength; positive
independent strengths determine an equivalent ratio. A positive alpha divides
the L1 threshold by alpha. Ridge and alpha-zero elastic net generally have no
finite threshold and use the L1 threshold divided by `0.001` as a finite path
anchor. `PenaltyMaximum` distinguishes this from a true zero-coefficient
threshold. SLOPE's threshold is the maximum cumulative sorted gradient magnitude
divided by the corresponding cumulative base weight. Axis penalties take the
maximum group threshold, preserving the `1/nobs` loss scale and explicit penalty
axis.

Coordinate paths cache column summaries, residual offsets, correlations, dual
certificate storage, and working-set masks across points. Gap Safe screening
uses the existing feasible dual vector `u` and radius `sqrt(2 * gap / n)`. A
zero coordinate is screened when its absolute dual correlation plus column norm
times the radius lies strictly below lambda, with conservative rounding guards.
Working sets include nonzero coefficients and KKT violators. Full checks after
sweeps restore excluded violators; residuals are refreshed before accepting
convergence and at iteration limits. Screening eligibility is rebuilt at each
strength and disabled at zero penalty. Single fits retain complete cyclic
sweeps. Proximal and multivariate paths reuse native products and iteration
storage with complete mapping checks, including all unpenalized intercepts.

### Dynamic Gap Safe zero screening

`Hybrid` and `ProximalGradient` enable zero screening by default for Gaussian
and binomial losses with `ClusteredLasso`, `ChainFusedLasso`, and `Slope`.
`.screening(false)` disables certificates and reductions. Optional
`GapSafeDatafit`, `GapSafePenalty`, and `ScreeningDesign` capabilities preserve
external components and runtime composition without additional solver bounds.
`Box` forwards capabilities. Other wrappers, multivariate layouts, and
product-only designs retain ordinary fitting. This does not add working sets,
fusion screening, Hybrid paths, or screening to the separable `Irls` solver.

Let $u$ be an observation-scale feasible dual point and $G$ its original-problem
gap. The dual has strong concavity $n$ for Gaussian loss and at least $4n$ for
binomial loss. The radii are therefore $\sqrt{2G/n}$ and $\sqrt{G/(2n)}$,
respectively, following the [Gap Safe
framework](https://www.jmlr.org/papers/v18/16-577.html). Gaussian candidates
start from residuals divided by $n$. Binomial candidates start from the negative
original-loss gradient. A free intercept requires $\mathbf1^\mathsf{T}u = 0$;
binomial candidates also require $(y_i - 1)/n \leq u_i \leq y_i/n$. A scalar
clipped-shift projection enforces the box and intercept constraint together.
Mean-centering alone would be invalid. Radial scaling then enforces the complete
penalty's polar constraints.

Clustered-lasso feasibility uses sorted signed subset sums. A subset of size $k$
has capacity $ak + bk(p - k)$ for L1 strength $a$ and fusion strength $b$. SLOPE
uses sorted absolute prefix sums bounded by cumulative rank weights. Chain
feasibility uses reachable edge-flow intervals for $g_j = s_j + t_{j-1} - t_j$,
with $|s_j| \leq a$, $|t_j| \leq c_j$, and endpoint flows zero. A bracketed
search retains a feasible scaling divisor. All checks verify correlations at the
actual scaled certificate, rather than assuming that floating-point products
commute with scaling.

After scaling, free-intercept certificates share a dyadic grid with a bounded
integer absolute sum. Integer balancing enforces the intercept equality exactly
for the stored values while retaining binomial box feasibility. Correlations are
then recomputed and checked again.

For zero L1 or entirely zero SLOPE weights, zero screening is inactive and the
certificate is the universally feasible $u = 0$. Its gap can remain loose even
at an optimum. For binomial proportions, $D(0)$ is mean response entropy, not
necessarily zero. No rank decision or nullspace projection is required for this
fallback. Convergence continues to use the original-space PG mapping.

Gaps use the loss Fenchel gap plus $P(\beta) - \beta^\mathsf{T}X^\mathsf{T}u$,
avoiding subtraction of complete objectives. Rounding allowances cover dot
products, normalization cancellation, and intercept equality. Norms use stable
accumulation; bounds for raw centered products also account for magnitudes
before cancellation. Directed rounding and strict inequalities retain uncertain
boundary features. These are floating-point numerical certificates, not an
interval-arithmetic proof. A materially negative gap is a numerical failure.

**Zero rules.** Write $g = X^\mathsf{T}u$ and $e_j = R\|x_j\|$ plus product
uncertainty. Clustered lasso certifies zero when $|g_j| + e_j < a - b(p - 1)$,
using the original feature count. The rule follows from bounding every incident
fusion subgradient; the ordinary lasso threshold is unsafe for this coupled
penalty. A nonpositive threshold certifies nothing.

For chain screening, forward reachable intervals $F_j$ describe $t_j$ using
nodes through $j$, and backward intervals $B_j$ describe it using later nodes.
The implementation propagates upper bounds on lower endpoints and lower bounds
on upper endpoints over the entire correlation-error box. With these robust
endpoints $F^L,F^U,B^L,B^U$, zero is certified only if

$$
  g_j + e_j - F^U_{j-1} + B^L_j < a,\qquad g_j - e_j - F^L_{j-1} + B^U_j > -a.
$$

For every possible optimal correlation, these inequalities provide a feasible
decomposition with an interior $s_j$. Fenchel support equality forces
$\beta_j = 0$ for every minimizer. This chain specialization follows from the
joint dual constraints; it does not assume that an interior decomposition at the
sphere center remains valid at the optimum.

SLOPE uses the complete prefix screening inequalities of [Elvira and
Herzet](https://arxiv.org/abs/2110.11784), with individual upper bounds
$U_j = |g_j| + e_j$. Sort remaining bounds decreasingly. Initialize $E = 0$ and
$A = \infty$; at rank $q$, save $A_q = \min(A,w_q + E)$, replace $A$ by $A_q$,
and set $E = \max(0,E + w_q - U_q)$. Starting at the smallest bound, remove the
tail while $U_r < A_r$. This recurrence evaluates the prefix/suffix tests in
linear time after sorting and supports repeated or trailing zero weights.
Exhaustive prefix tests independently validate the optimized recurrence.

**Exact restrictions.** Only currently zero coefficients receive new fixed-zero
constraints, preserving accepted objective descent. Removing $m_0$ clustered
lasso zeros changes surviving L1 strength to $a + bm_0$. Retain all fusion terms
between survivors. SLOPE retains the first $m$ rank weights for $m$ survivors.
Chain removal splits active runs and adds each edge touching a removed zero to
its surviving endpoint's unary L1 weight. Never join across a removed node. For
weighted TV $T$ and nonnegative node weights $\alpha$, the exact prox is

$$
  \operatorname{prox}_{\tau(T+\sum_j\alpha_j|\cdot_j|)}(v)
  = [\operatorname{prox}_{\tau T}(v - \tau\alpha)]_+ + [\operatorname{prox}_{\tau T}(v + \tau\alpha)]_-.
$$

TV's prox preserves order, and clipping preserves its edge subgradients. The
positive and negative constrained minimizers therefore have disjoint supports.
Both the quadratic objective and TV split over those supports, proving the
formula. Two existing taut-string calls suffice; uniform weights retain the
one-call fast path. This restriction is separate from `Masked`, which excludes
coordinates from penalization instead of fixing them to zero.

Screening runs initially and every ten PG updates, at Gaussian Hybrid checks,
and at accepted logistic outer iterates. Logistic surrogate gaps never establish
restrictions. Selected column products and aggregate rebuilds skip screened
columns. A zero block containing certified features stays fixed during Hybrid
sweeps; PG may split and reactivate its unscreened members. Stopping and final
labels remain in the original space, and observers never trigger certificates.
Every existing proximal path point resets restrictions while retaining scratch
capacity and warm-start coefficients. Thus features can reactivate as strengths
change. Returned vectors always retain the original feature dimension.

Fit diagnostics separate certificate passes, zero tests, certified features,
active dimensions, rebuilds, and fallback certificates. Optional gaps do not
change the stopping rule. `task bench:screening` compares complete enabled and
disabled solves at matched independently checked original-space accuracy,
including certificates, allocations, and peak memory. Speedups are not assumed.

IRLS paths reuse preprocessing, weighted summaries, residual buffers, and
working-set storage across strengths. Working sets apply to both standalone fits
and paths; every inner sweep checks the complete quadratic KKT conditions and
restores excluded violators. Eligibility is rebuilt for every quadratic model,
and only a full original-loss check certifies convergence. IRLS does not provide
safe screening or a dual certificate. Automatic paths reuse `NullModelDatafit`
and `PathPenalty`, including binomial null-intercept errors and ridge path
anchors. Standalone fits and explicit paths start at zero; automatic paths start
at the optimized null intercept.

IRLS workspace is linear in observations and features, in addition to stored
path results. LazyMatrix 0.6.0 computes each weighted column norm in
`O(nobs + nnz)` time, including implicit zeros. These norms are cached once per
outer iteration. Noncanonical sparse columns can require temporary linear
scratch inside LazyMatrix. Sparse residual updates preserve centering through a
weighted offset rather than visiting implicit zeros for each coordinate.

## 9. Out-of-core execution

Develop support in explicit stages:

1. File-backed `X`, with coefficients and required `O(n)`/`O(p)` workspaces in
   RAM.
2. File-backed `X` with streamed or file-backed `O(n)` workspaces; coefficients
   and essential `O(p)` state remain in RAM.
3. Out-of-core coefficients are deferred.

Record efficient storage orientation, not merely whether random access exists.
Row storage favors streaming; column storage favors coordinate methods. Working
sets may cache active columns, with periodic full scans for optimality checks.
Sorting-dependent penalties such as SLOPE require global coefficient operations.

Introduce an execution policy when needed: memory budget, block size, thread
count, cache policy, and permitted temporary storage. Transposes, format
conversions, dense copies, and disk caches require explicit permission through
that policy. No hidden full-matrix materialization. Report data passes and I/O
where measurable; document whether a memory budget is hard or advisory.

Keep file-format and compression dependencies optional and outside lean solvers.

## 10. Future GPU backend

GPU support is deferred implementation, not an architectural exclusion.
High-level model/penalty specifications, layouts, and solver configuration must
remain independent of host memory. Do not make `Vec<f64>` the universal public
coefficient representation.

CPU oracles can use CPU-specific buffers. Future GPU oracles and solvers may use
separate device buffers and kernels, with matrices, coefficients, gradients, and
proximal workspaces resident on the device between iterations. Allow explicit
transfers and future asynchronous execution without requiring an async API now.

Bulk products, reductions, matrix coefficients, and proximal/splitting methods
are initial accelerator candidates. GPU dependencies belong in optional backend
crates; do not require every CPU coordinate kernel to become device-generic.

## 11. Workspace and public APIs

Start with one `shrinkage` library crate and optional backend features. Use
modules within that crate until dependency isolation or reuse justifies a
workspace split. Use `name.rs` and `name/child.rs` for modules; do not create
`mod.rs` files. Possible later boundaries are:

  | Crate/component                 | Responsibility                                               |
  | ------------------------------- | ------------------------------------------------------------ |
  | `shrinkage-core`                | Problem, layout, capability, error, and result types         |
  | `shrinkage-models`              | Built-in datafits and metadata                               |
  | `shrinkage-penalties`           | General penalties and composition                            |
  | `shrinkage-cd`                  | Lean coordinate/block solvers and lasso entry point          |
  | `shrinkage-prox`                | Generic proximal and splitting solvers                       |
  | `shrinkage-slope`               | SLOPE penalty machinery and specialized solver               |
  | `shrinkage`                     | Convenience API, runtime facade, solver selection, utilities |
  | `bindings/python`, `bindings/r` | Language interfaces                                          |

Do not create all these as empty crates. No leaf solver may depend on the
batteries-included facade. Keep LazyMatrix external. Generic solvers should not
depend on the complete catalog of built-in models and penalties.

Provide a simple lasso convenience API and a typed compositional API built on
the same implementation. The dynamic API exposes conventional fit/path/predict
operations and clear compatibility errors. Language callbacks must not occur per
observation or coefficient. Zero-copy inputs require validated layout, lifetime,
mutability, and ownership contracts.

### Python dense input boundary

The optional binding package in `packages/python` remains a separate Cargo
workspace. NumPy is a Python runtime dependency; its input adapter uses PyO3's
buffer API without adding Python or matrix backend dependencies to the default
Rust build. The binding enables the versioned ndarray backend in its own
workspace, moving each copied design vector into an owned column-major ndarray
without another matrix copy. LazyMatrix supplies its raw columns and statistics.
The first numerical boundary is dense, single-response Gaussian data. The
private `_prepare_dense_inputs` entry point tests the shared `DenseInputs`
container independently of fitting.

Accept exact `numpy.ndarray` instances with native-endian `float64` dtype and
aligned storage. Require a two-dimensional design of shape `(n, p)` and a
one-dimensional response of length `n`, with `n > 0` and `p >= 0`. Reject
nonfinite values. Do not squeeze, cast, or invoke array protocols implicitly.
Reject subclasses, including masked arrays, so user indexing and conversion
hooks cannot enter the native boundary. Type and dtype errors become
`TypeError`; shape, alignment, value, and copying-policy errors become
`ValueError`. The Python package guide documents explicit conversions.

Every NumPy input is copied once into immutable Rust-owned storage: the design
uses column order, and the response uses a vector. Accept C, Fortran,
transposed, sliced, reversed, and zero-stride broadcast layouts by copying
logical values through their strides. Treat owned arrays and views, and writable
and read-only arrays, identically. No NumPy input qualifies for zero-copy
access; reject `copy=False`. Read-only flags alone cannot establish exclusive
ownership because writable aliases can exist. This policy costs
`8 * (n * p + n)` bytes beyond the input arrays, excluding allocation metadata.

Keep the GIL and buffer exports alive throughout copying. Callers must prevent
concurrent mutation or resizing, including through aliases or native operations
already running without the GIL, until preparation returns. Release all buffer
exports before any detached numerical operation. The resulting `DenseInputs`
owns only a Rust ndarray and response vector and retains no Python objects.
Tests exercise detached reads, source collection, alias mutation after
preparation, input flags, error cleanup, and all accepted layouts. Fitting and
prediction consume this ownership boundary while releasing the GIL around Rust
numerical calls. Prediction accepts empty batches and applies the same design
validation. Because the public methods do not signal completion of copying,
callers keep inputs unchanged until the call returns. No scalar loop calls
Python, including during layout copying and finite-value validation.

The Python `Lasso` configuration returns an immutable `LassoFit` wrapper over
the Rust result. It exposes all normalization presets, intercept selection,
sweep budgets, and explicit `StoppingCriterion` values for duality gap or KKT
violation. Defaults resolve in Rust. Coefficients, preprocessing, and
diagnostics retain Rust's statistical conventions. Array-valued properties
return independent NumPy copies; the fit retains no training arrays or design
snapshots. Prediction uses the native original-scale parameters without
estimating new preprocessing.

Map Rust invalid-input and unsupported-combination errors to `ValueError`, and
nonfinite arithmetic to `FloatingPointError` with the operation and iteration.
The selected dense backend is infallible, so no backend failure can arise in
this slice. Return finite iteration-limit results with `"iteration_limit"`
termination and their final diagnostics, without labeling them converged.
Committed fixtures generated by `examples/python_lasso_fixture.rs` cover all
normalization presets with and without intercepts, both stopping criteria, zero
penalty, an initially optimal fit, and an iteration limit. The generator's
`--check` mode guards against stale fixtures. Installed wheel and isolated
source-distribution tests compare all exposed diagnostics and predictions,
alongside analytical fits and independent objective and KKT calculations.

## 12. Results, correctness, and performance

Return coefficients, intercept/preprocessing metadata, termination reason,
iteration count, and applicable diagnostics. Separate invalid/unsupported input
errors from optimization termination. Reaching an iteration limit is not
success.

Document each tolerance: KKT violation, duality gap, proximal-gradient mapping,
fixed-point residual, or another justified criterion. Do not fabricate a duality
gap for objectives lacking an implemented valid dual. Include solver selection,
objective conventions, and available work counters in diagnostics.

The convergence API is `.terminate_on(StoppingCriterion)`. Each criterion owns
its tolerances, so changing criteria cannot silently reuse a tolerance with a
different meaning. Provide concise constructors for common choices and explicit
absolute and relative tolerances where applicable. Illustrative calls are:

```rust,ignore
let model = Lasso::new(0.1)
    .terminate_on(StoppingCriterion::duality_gap(1e-6))
    .max_iterations(10_000);

let model = Lasso::new(0.1).terminate_on(StoppingCriterion::DualityGap {
    absolute: 1e-10,
    relative: 1e-6,
});

let model = Lasso::new(0.1)
    .terminate_on(StoppingCriterion::kkt_violation(1e-8));
```

`duality_gap(tol)` denotes a relative tolerance; `kkt_violation(tol)` denotes an
absolute tolerance. Define the gap threshold as `absolute + relative * scale`,
with a documented reference scale for each supported problem. For Gaussian
lasso, use the averaged squared loss at the zero-coefficient model, with the
intercept optimized when enabled. Keep this scale fixed during a fit, and define
zero-scale behavior without dividing by it. Validate finite, nonnegative
tolerances with at least one positive component.

Duality gap is the default for positive-penalty Gaussian lasso. The
`.tolerance(...)` API still controls absolute KKT violation. A feasible dual
vector `u` satisfies `||X_tilde' u||_inf <= lambda` and, when fitting an
intercept, `1' u = 0`. Its objective is `y' u - n ||u||² / 2`. The solver
centers `r/n` when fitting an intercept and scales it to satisfy the column
constraints. At zero penalty this construction can yield the zero vector and a
loose gap, so the default uses KKT violation. The gap is evaluated using a
Fenchel decomposition to avoid subtracting nearly equal objectives. Unsupported
criteria must be rejected before iteration, without silently substituting
another criterion. Nonconvex models require an appropriate stationarity
criterion.

Iteration limits remain independent budgets and report `IterationLimit`, not
convergence. Report the selected criterion, its final value, and its threshold.
Keep diagnostic computation separate from criterion selection: a fit may report
both KKT violation and a duality gap. A cheap check can trigger a more expensive
certificate calculation, but only the requested certificate may establish
convergence. Check frequency is a separate solver policy, and final convergence
checks must use refreshed state.

Verification must cover:

- Analytical solutions, finite-difference derivatives, and proximal optimality.
- Dense/sparse and explicit/lazy normalization equivalence.
- Prediction-preserving coefficient transformations and intercept treatment.
- Typed/dynamic and in-memory/file-backed agreement.
- KKT or stationarity checks and independent reference fixtures.
- Warm/cold starts, paths, invalid inputs, numerical failure, and
  nonconvergence.
- Nonconvex initialization sensitivity without assuming identical local
  solutions.

Benchmarks measure time, allocations, memory, matrix products, and data passes,
with I/O for file-backed cases. Reuse workspaces and avoid hot-loop allocation.
Never form a Gram matrix by default. Performance complications require measured
benefit; test mathematical correctness before optimizing.

## 13. Bootstrap and implementation milestones

1. **Bootstrap:** one Cargo library crate, README, this design, license,
   formatting, linting, CI, and reproducible development tooling. Pin the
   inspected LazyMatrix release and verify backend integration with a runnable
   example. Add solver tests and the benchmark harness with the first numerical
   implementation. Keep bindings and GPU dependencies out.
2. **First vertical slice:** Gaussian lasso coordinate descent with an
   unpenalized intercept, dense and sparse CSC input, optional lazy
   normalization, original-scale predictions, and a meaningful convergence
   check. Verify against analytical cases and reference fixtures; benchmark a
   lean consumer. Verify the normalized-coefficient penalty convention,
   `O(nnz_j)` sparse coordinate work, and residual refreshes explicitly.
3. **Composition proof:** extract minimal capabilities; add proximal gradient,
   ridge/elastic net, and an object-safe oracle. Demonstrate the same algorithm
   with concrete oracles and runtime problems whose components are selected
   independently. Measure normalized-product allocations and establish workspace
   ownership. Complete the bounded block-reader prototype and verify injected
   failures through preprocessing and fitting before stabilizing these APIs.
4. **Datafit and layout proof:** add binomial regression through proximal
   gradient before expanding the solver catalog. Test stable loss/derivative
   evaluation, intercept handling, and step-size selection. Add a small
   multivariate Gaussian example with explicit penalty axes before stabilizing
   coefficient layouts; compare separable cases with independent response fits.
5. **Paths and penalty foundations:** the exact SLOPE prox, warm-start paths,
   lasso screening, maximum penalties, scaled and masked penalties, row
   composition, and validated disjoint group definitions are implemented.
6. **First Python slice:** establish packaging and dense input ownership, then
   expose Gaussian lasso fitting, prediction, and diagnostics. Keep the Rust
   default build lean and test the installed package early.
7. **First-release penalties:** disjoint Group LASSO has an exact proximal map
   and independent Gaussian reference fits. Add uniform clustered lasso with an
   exact complete proximal map and Gaussian reference fits. Add the dense hybrid
   of PG and cluster coordinate descent, signed cluster state, full-space
   stopping checks, normalization, intercepts, and paths. Test the conversion
   from the research project's unaveraged loss to Shrinkage's average loss.
8. **Python release surface:** expose dense Gaussian fitting, paths, prediction,
   and explicit solver selection for lasso, ridge, elastic net, Group LASSO, and
   clustered lasso. Publish the tested compatibility matrix and verify parity
   with Rust, including the hybrid method.
9. **Publication and paper integration:** test distributable Rust and Python
   artifacts, reproduce a frozen research example through the installed package,
   and publish versioned packages and citation metadata. Pin the published
   implementation in the paper's reproduction instructions.

After the first release, consider broader penalties and solvers, additional
outcomes, out-of-core execution, R bindings, cross-validation, broader Python
coverage, automatic selection, and optional accelerators. The backlog in
[TODO.md](TODO.md) retains their scoped tasks.

Each milestone must leave a usable, tested subset. Broad advertised capability
must not get ahead of implemented and verified combinations.

## 14. Guidance for coding agents

- Preserve the lean Rust use case and runtime-configurable package use case.
- Reuse LazyMatrix; extend it for general matrix capabilities when necessary.
- Derive small traits from working algorithms rather than anticipated features.
- Keep solver-specific machinery out of unrelated component interfaces.
- Validate mathematical assumptions, shapes, parameter ranges, and storage
  needs.
- Do not infer a prox for a sum or convergence guarantees from component names.
- Make allocation, preprocessing, loss scaling, and device transfers explicit.
- Resolve ownership, exact view types, multinomial conventions, and initial file
  formats through concrete implementations. Preserve the initial normalization
  and penalty-scale conventions above.
- Record significant architectural decisions and deviations in this file.

## Solver observations for documentation

Single typed fits may implement `ObservedSolver` in addition to `Solver`.
`Problem::fit_with_observer` exposes borrowed optimization-scale coefficients,
the intercept, the iteration count, and the accepted step kind. Observers cannot
change numerical state, acceptance, or stopping. Ordinary fits use the same
kernels with a statically dispatched no-op observer, without trace storage.
Coordinate descent reports coordinate and intercept updates. Proximal gradient
reports accepted full updates. IRLS reports accepted outer updates. Gaussian
hybrid reports proximal transitions and complete cluster sweeps; binomial hybrid
reports accepted outer updates on the original objective and retains its
cumulative inner iteration count. Observers may also implement `observe_inner`
to receive IRLS coordinate updates, hybrid block updates, and binomial hybrid
quadratic steps. These callbacks include the parent outer iteration and the
inner iteration count. Surrogate states need not descend the original objective
or become accepted outer states. Closure observers ignore them by default, and
rejected line-search trials remain omitted. Initialization and successful final
states are always observed, including initially optimal and iteration-limited
fits. Paths and erased runtime fits retain their existing interfaces.

The documentation explorer lives in a separate Cargo workspace and compiles
these library implementations to WebAssembly. It fixes the intercept at zero and
uses standardized two-column designs without additional normalization, so
plotted coordinates equal the observed and returned coefficients. Loss, penalty,
and objective surfaces use the same Rust components as fitting. A tighter
proximal fit supplies the reference optimum independently of solver selection.

`InitialPoint` and `Problem::fit_from_with_observer` initialize a single
observed fit from finite optimization-scale coefficients and an intercept. The
coefficient count must match the design; a disabled intercept requires zero.
Solvers rebuild residuals and other numerical state from these parameters.
Ordinary observed fits retain each solver's default initialization.
