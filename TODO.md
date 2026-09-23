# Shrinkage roadmap

This checklist tracks implementation of [DESIGN.md](DESIGN.md). The repository
currently fits Gaussian lasso models through a convenience API. Work through the
milestones in order, with each leaving a usable, tested subset. Check off
numerical features only after their correctness tests and relevant benchmarks
are in place.

Keep architectural decisions and statistical conventions in `DESIGN.md`, and
update this checklist as work lands. Start with `f64` and modules in the
existing crate. Preserve the lean default build, and add dependencies or crate
boundaries only when an implemented consumer needs them.

## 1. Bootstrap

- [x] Create the Rust 2024 library with a declared Rust 1.89 MSRV, README,
  design, and license.
- [x] Pin LazyMatrix with optional `faer`, `nalgebra`, `ndarray`, and `sprs`
  backends, versioned features, short aliases, and no default matrix
  backend.
- [x] Add a runnable example with dense and CSC input that verifies lazy
  normalization and prediction-preserving coefficient and intercept
  transformations.
- [x] Configure reproducible development tooling, formatting, linting, and CI.

## 2. First vertical slice: Gaussian lasso

- [x] Implement Gaussian lasso coordinate descent with an unpenalized intercept
  and a simple convenience API, using LazyMatrix's dense and sparse CSC
  column capabilities.
- [x] Apply the loss and penalty conventions in [Design
  §8](DESIGN.md#8-statistical-semantics): average squared loss with the
  `1/(2n)` factor and L1 penalization of optimization-scale coefficients.
- [x] Fit training-column means and population standard deviations by default.
  Under `Normalization::Auto`, disable centering when the intercept is
  disabled, and retain fitted preprocessing metadata.
- [x] Replace `standardize(bool)` with `normalize(Normalization)`. Provide
  common presets and custom independent centering and scaling using
  LazyMatrix's enums. Honor explicit choices independently of intercept
  fitting, preserve induced intercepts, and test dense/sparse and
  explicit/lazy equivalence.
- [ ] Validate dimensions, nonempty observations, finite inputs, penalty
  strengths, and explicit scales before constructing LazyMatrix views.
  Replace computed zero scales with one and hold zero-norm normalized
  columns at zero. Convenience-API validation is implemented; user-supplied
  center and scale vectors remain pending.
- [x] Implement sparse residual state with a scalar centering offset, cached
  base sum, and column summaries. Keep coordinate updates and cached-sum
  column dots at `O(nnz_j)`, and periodically refresh residuals to control
  drift.
- [ ] Return original-scale coefficients and predictions while preserving any
  intercept induced by preprocessing, including through the typed API. The
  convenience API is implemented; the compositional typed API remains
  pending.
- [x] Add a documented KKT convergence check and report termination reason,
  iterations, and objective diagnostics. Distinguish iteration limits,
  numerical failures, and invalid input from convergence.
- [ ] Implement a valid Gaussian lasso dual certificate and duality gap. Test
  feasibility, objective scaling, intercept and normalization policies, zero
  penalty, and zero reference loss before making relative duality gap the
  default convergence criterion.
- [ ] Introduce `terminate_on(StoppingCriterion)` with criterion-specific
  tolerances, concise duality-gap and KKT constructors, and explicit
  absolute and relative gap tolerances. Document the reference scale and
  migrate the existing absolute-KKT `tolerance` setter without silently
  changing its meaning. Keep iteration budgets independent, reject
  unsupported criteria, and report the selected criterion, final value, and
  threshold.
- [ ] Verify analytical cases, independent reference fixtures, agreement between
  dense and sparse fits, and agreement between explicit and lazy
  normalization. Test intercept policies, constant columns, penalty scaling,
  invalid inputs, and nonconvergence. Analytical and independently solved
  small cases are covered; external reference fixtures remain pending.
- [ ] Add the benchmark harness and a lean consumer benchmark. Test sparse
  operation counts and reconstructed residuals, and measure time,
  allocations, and memory without forming a Gram matrix by default. Timing
  benchmarks, operation counts, and residual tests are implemented;
  allocation and memory profiling remain pending.

## 3. Composition proof

- [ ] Extract small problem, datafit, penalty, result, and solver capabilities
  from the working lasso implementation. Keep matrix operations in
  LazyMatrix and solver state in Shrinkage.
- [ ] Add proximal gradient, ridge (squared L2), and elastic net. Require a
  valid proximal map for the complete nonsmooth term.
- [ ] Run the same iteration routine with concrete and object-safe oracles.
  Construct runtime problems by independently selecting the design operator,
  predictor datafit, and complete proximal term; keep dispatch outside
  scalar loops and generic column views inside concrete adapters.
- [ ] Compare typed and runtime fits and return actionable errors for
  unsupported combinations.
- [ ] Establish reusable workspace ownership across iterations and path points.
  Measure normalized-product and proximal allocations, including LazyMatrix
  0.3.0's scaled input clone. Verify reusable normalization scratch storage
  upstream before claiming allocation-free normalized iterations.
- [ ] Prototype a separate fallible block-reader capability with bounded buffer
  reuse and explicit borrowing contracts before stabilizing solver
  interfaces.
- [ ] Propagate backend errors with their source and operation context. Inject
  read failures during preprocessing and a later solver iteration, and
  verify that partially written outputs are neither consumed nor reported as
  success.

## 4. Datafit and coefficient layout proof

- [ ] Add binomial regression through proximal gradient with stable loss and
  derivative evaluation, intercept handling, and justified step-size
  selection.
- [ ] Check derivatives with finite differences and verify fits against
  independent reference cases, including extreme linear predictors.
- [ ] Add multivariate Gaussian fitting with explicit feature, response, and
  penalty axes before stabilizing coefficient layouts.
- [ ] Compare separable multivariate cases with independent response fits and
  provide a small runnable example.

## 5. SLOPE and regularization paths

- [ ] Implement SLOPE's exact proximal map, validate nonnegative, nonincreasing
  weights, and test proximal optimality.
- [ ] Add the hybrid coordinate/proximal solver with SLOPE-specific ordering,
  signs, equal-magnitude clusters, updates, and merges. Use full proximal
  steps to permit cluster splitting.
- [ ] Test ties, cluster merges and splits, normalization semantics, and
  agreement with generic proximal methods before preferring the hybrid
  solver.
- [ ] Add path APIs with warm starts and reusable workspaces. Add screening and
  working sets with full optimality checks at each solution.
- [ ] Implement maximum-penalty calculations and path sequences for each
  supported family and penalty. Keep independent penalty strengths explicit
  when tuning multiple penalties.
- [ ] Compare warm and cold starts and benchmark paths and SLOPE solvers across
  representative dense and sparse problems.

## 6. Broader penalties and solvers

- [ ] Add scaled penalties, explicit unpenalized masks, and composition over
  rows and groups. Validate group definitions and distinguish disjoint from
  overlapping groups.
- [ ] Add group and sparse-group penalties with valid proximal or block
  subproblems, followed by group SLOPE, nuclear norm, bounds, and
  constraints where supported.
- [ ] Add FISTA, three-operator splitting, and ATOS with documented assumptions.
  Do not substitute successive proximal maps for the prox of an arbitrary
  sum.
- [ ] Add ADMM through explicit splittings and supported subproblem solvers.
- [ ] Add MCP, SCAD, and group variants with suitable coordinate, block,
  majorization-minimization, or difference-of-convex methods. Document
  initialization sensitivity and justified stationarity criteria separately
  from convex convergence guarantees.
- [ ] Test proximal and subproblem optimality and solver agreement where
  solutions are comparable. Benchmark specializations before adding
  selection preferences.

## 7. Additional outcomes and statistical semantics

- [ ] Add Poisson and other GLMs with validated observation weights and offsets.
- [ ] Choose and document multinomial identifiability and penalty conventions,
  then implement multinomial fitting.
- [ ] Extend multivariate models and penalties with explicit coefficient layouts
  and penalty axes.
- [ ] Add Cox and stratified Cox with event, time, and stratum validation and
  risk-set operations in the datafit or solver. Preserve observation
  coupling in any blockwise evaluation.
- [ ] Add proximal Newton, IRLS, and other specialized methods when concrete
  datafits justify their curvature or weighted-product capabilities.
- [ ] Verify each family's loss normalization, derivatives, weight and offset
  semantics, intercept policy, and reference fits.

## 8. Out-of-core execution

- [ ] Build a reference file-backed adapter on the tested fallible block
  contract, initially retaining coefficients and required `O(n)` and `O(p)`
  state in RAM. Keep format and compression dependencies optional.
- [ ] Add streaming kernels for compatible datafits and record efficient row or
  column access, rather than assuming arbitrary access is inexpensive.
- [ ] Define an execution policy for memory, block size, threads, caches, and
  permitted temporary storage. Require explicit policy permission for
  copies, transposes, conversions, and disk caches; document hard versus
  advisory limits.
- [ ] Add active-column caching and periodic full optimality scans where useful.
- [ ] Extend to streamed or file-backed `O(n)` workspaces while retaining
  coefficients and essential `O(p)` state in RAM.
- [ ] Compare in-memory and file-backed fits, and verify bounded buffers, read
  failures, and absence of hidden full-matrix materialization. Measure data
  passes, I/O, memory, and time.

## 9. Bindings, selection, and model evaluation

- [ ] Expose runtime specifications for fitting, paths, and prediction backed by
  the typed implementations and a tested compatibility matrix.
- [ ] Implement a documented deterministic `auto` policy based on mathematical
  compatibility, storage access, memory, and available implementations.
  Report the chosen solver and permit overrides.
- [ ] Add cross-validation with preprocessing fitted separately on each training
  fold and reused across that fold's path.
- [ ] Add R and Python bindings with validated ownership, lifetime, layout, and
  mutability contracts for zero-copy inputs. Keep language callbacks outside
  observation and coefficient loops.
- [ ] Compare typed fits, runtime fits, and binding results. Verify prediction
  transformations, error propagation, and cross-validation without
  preprocessing leakage.
- [ ] Document supported combinations, objective and tolerance conventions,
  diagnostics, examples, and measured performance before advertising
  coverage.

## 10. Optional accelerators

- [ ] Select representative bulk workloads after CPU interfaces are proven.
- [ ] Add optional device-specific operators, buffers, oracles, and kernels
  without making host slices or `Vec<f64>` universal public requirements.
- [ ] Keep matrices and optimization workspaces on the device between iterations
  and make transfers explicit. Isolate GPU dependencies from lean CPU
  solvers.
- [ ] Verify numerical agreement between CPU and device fits and measure
  transfer costs, memory, and end-to-end performance.

## Deferred work

- [ ] Support original-scale penalties under nontrivial normalization only after
  implementing their transformed subproblems explicitly.
- [ ] Consider splitting the project into a workspace of crates when dependency
  isolation or reuse justifies them, and a more sophisticated selection
  planner when measurements justify one.

Distributed execution, general symbolic optimization, out-of-core coefficient
vectors, and global optimization of arbitrary nonconvex objectives remain
outside the initial scope.
