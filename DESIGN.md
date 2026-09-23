# Shrinkage: design

Status: Gaussian lasso convenience API implemented; composition is the next
architectural milestone. This document records architectural commitments and an
implementation sequence; illustrative API names are not frozen interfaces.

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

  | Component | Intended coverage                                                                                                                                                                   |
  | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Datafits  | Gaussian, multivariate Gaussian, binomial, multinomial, Poisson, other GLMs, Cox and stratified Cox                                                                                 |
  | Penalties | L1, squared L2/ridge, elastic net, SLOPE, group and sparse-group penalties, group SLOPE, MCP, SCAD, group variants, other nonconvex penalties, nuclear norm, bounds and constraints |
  | Solvers   | Coordinate and block descent, hybrid SLOPE descent, gradient and proximal-gradient methods, FISTA, three-operator splitting and ATOS, ADMM, proximal Newton/IRLS, MM/DC methods     |
  | Storage   | Dense, sparse, memory-mapped, chunked/file-backed, and user-defined operators                                                                                                       |
  | Utilities | Regularization paths, warm starts, screening, prediction, cross-validation, diagnostics                                                                                             |

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

Build working numerical code before stabilizing abstractions. Do not introduce
an elaborate type system or one crate per prospective feature at bootstrap.

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

For the initial CPU proximal solver, erase the design operator, predictor
datafit, and complete proximal term independently when constructing a runtime
problem. Each adapter retains its concrete backend and workspace internally.
Erasing a fully instantiated `Problem<Matrix, Datafit, Penalty>` afterward does
not avoid constructing the Cartesian product. The same iteration routine must
accept concrete and erased solver oracles, with no runtime dispatch inside
scalar loops. Specialized coordinate solvers may erase a complete sweep.

LazyMatrix's `Columns` and `RawColumns` use generic associated view types, and
`LogicalColumn` has generic methods. They are static capabilities, not runtime
trait-object interfaces. Keep them inside concrete adapters. The composition
milestone must demonstrate runtime construction from independently selected
operators, datafits, and penalties as well as agreement with typed fits.

An `auto` policy considers mathematical compatibility, storage access, memory,
and available implementations. Initially use a documented deterministic policy;
defer a sophisticated planner. Report the chosen solver and allow overrides.
Unsupported combinations must produce actionable errors.

## 5. Matrices and LazyMatrix

Use `lazymatrix` as the foundation for matrix operations and just-in-time
normalization. Reuse its existing capabilities and backend adapters, including
dense and sparse integrations, rather than defining a competing matrix API.
Inspect dependency upgrades before changing the pinned release or relying on new
capabilities.

The crate pins the inspected release `lazymatrix = "=0.3.0"`, with default
features disabled. Shrinkage's versioned features `faer_v0_24`,
`nalgebra_v0_34`, `ndarray_v0_17`, and `sprs_v0_11` forward to the matching
LazyMatrix features. The short names `faer`, `nalgebra`, `ndarray`, and `sprs`
remain aliases for these release lines. Gate tests, doctests, examples, and
benchmarks on the versioned features so either entry point works. These retain
faer 0.24, nalgebra 0.34 with nalgebra-sparse 0.11, ndarray 0.17, and sprs 0.11.
ndarray arrays and borrowed views support row-major, column-major, and strided
storage. sprs CSC matrices and views use LazyMatrix's checked `SprsCsc` wrapper;
callers explicitly convert CSR input to CSC before fitting. Version 0.3.0
supplies fallible statistics and products, including a combined
normalization-statistics hook. Other backends and storage capabilities remain
outside this solver's public feature set. Keep upstream development overrides
local so a clean checkout builds without the sibling repository.

Expose only one release line per backend until LazyMatrix supports concurrent
implementations for multiple versions. In release 0.3.0, Cargo feature
unification can enable a newer adapter through another dependency and remove the
trait implementations for older matrix types. Versioned feature names do not
isolate a consumer from this upstream limitation.

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

Reusable output does not imply allocation-free execution. In release 0.3.0,
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
LazyMatrix 0.3.0. The lasso uses the combined `normalization_stats` hook and
preserves preprocessing errors with their operation context and source type.
Borrowed column views remain infallible and must not hide I/O or decoding. A
fallible block reader still needs a separate capability. Establish its borrowing
and buffer-reuse contracts during the composition milestone, before solver
interfaces stabilize. The prototype must use a bounded block buffer and an
injected read failure, including during preprocessing and a later solver
iteration. Full file-format support remains a later milestone.

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

## 7. Specialized SLOPE support

Implement SLOPE both as a generic penalty with an exact proximal map and through
the specialized hybrid coordinate/proximal solver. Validate nonnegative,
nonincreasing SLOPE weights for the convex penalty.

The specialized solver needs ordering, signs, equal-magnitude clusters, cluster
updates/merges, and full proximal steps capable of splitting clusters. Put this
machinery behind a SLOPE-specific interface; do not force it into every penalty.
Use generic proximal methods as independent correctness references and
fallbacks. Solver selection should prefer specialized methods only where
compatibility and benchmarks justify it.

## 8. Statistical semantics

Coefficient layouts distinguish features, responses, classes, and groups even
when internal storage is flattened. Penalty axes must be explicit. Multinomial
identifiability conventions must be chosen and documented because
reference-class and symmetric representations change penalty semantics.

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
records the resolved centering and scaling rules and values. User-supplied
center and scale vectors remain later work.

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

The implemented `Lasso` builder defaults to an absolute KKT tolerance of `1e-6`
and at most 10,000 complete cyclic sweeps. For `c_j = X_tilde_j' r / n`, use
`|c_j - lambda sign(theta_j)|` on active coordinates and
`max(|c_j| - lambda, 0)` on zero coordinates, together with `|mean(r)|` when
fitting an intercept. Check the maximum violation after each sweep. Reconstruct
residuals every 50 sweeps, at the iteration limit, and before accepting
convergence. An initially optimal fit takes zero sweeps.

`LassoFit` owns original-scale parameters, fitted centers and scales, and final
objective and KKT diagnostics. A finite fit that exhausts its budget returns
`Termination::IterationLimit`. Invalid inputs, nonfinite arithmetic, and backend
preprocessing errors return distinct `LassoError` variants. User-supplied center
and scale vectors and the compositional typed API remain later work.

Paths support warm starts, reusable working sets, and full optimality checks.
Automatic maximum-penalty calculations and sequences are
family/penalty-specific. For multiple penalties, allow independent strengths and
meaningful mixing parameterizations without assuming every tuning problem is
one-dimensional.

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

## 12. Results, correctness, and performance

Return coefficients, intercept/preprocessing metadata, termination reason,
iteration count, and applicable diagnostics. Separate invalid/unsupported input
errors from optimization termination. Reaching an iteration limit is not
success.

Document each tolerance: KKT violation, duality gap, proximal-gradient mapping,
fixed-point residual, or another justified criterion. Do not fabricate a duality
gap for objectives lacking an implemented valid dual. Include solver selection,
objective conventions, and available work counters in diagnostics.

The planned convergence API is `.terminate_on(StoppingCriterion)`. Each
criterion owns its tolerances, so changing criteria cannot silently reuse a
tolerance with a different meaning. Provide concise constructors for common
choices and explicit absolute and relative tolerances where applicable.
Illustrative calls are:

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

Duality gap is the intended default for Gaussian lasso and other supported
convex problems once valid dual certificates are implemented and tested. The
current `.tolerance(...)` API still controls absolute KKT violation; do not
silently reinterpret it as a gap tolerance. Introduce `terminate_on` when
implementing the criteria, with an explicit migration from the existing setter.
Unsupported criteria must be rejected before iteration, without silently
substituting another criterion. Nonconvex models require an appropriate
stationarity criterion.

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
5. **SLOPE and paths:** exact prox, hybrid solver, cluster tests, warm starts,
   path state, and comparison with generic proximal methods.
6. **Broader objectives:** group/sparse-group penalties, MCP/SCAD, suitable
   specialized solvers, FISTA, TOS/ATOS, and ADMM with explicit assumptions.
7. **Additional outcomes:** Poisson, multinomial, broader multivariate models,
   and Cox; validate weights, offsets, and identifiability.
8. **Out-of-core:** build on the tested fallible block contract with a reference
   file-backed adapter, streaming kernels, budgets, and working-set caching.
9. **Bindings and selection:** R/Python runtime specs, compatibility reporting,
   paths/prediction/CV, and documented automatic selection.
10. **Accelerators:** optional device-specific backend after CPU interfaces and
    representative bulk workloads are established.

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
