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

Each unchecked item below is intended for one focused `/goal` session. Copy the
whole item, including its scope and checks, when starting a goal. Complete named
prerequisites first. Numerical implementation items include correctness tests,
relevant benchmarks, and documentation for the stated combination; broader
coverage belongs in a separate item. Where a design choice is still open, the
item produces a decision and concrete follow-up tasks.

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
- [x] Validate dimensions, nonempty observations, finite inputs, penalty
  strengths, and explicit scales before constructing LazyMatrix views.
  Replace computed zero scales with one and hold zero-norm normalized
  columns at zero. Supplied center and scale vectors override their
  respective rules and are validated before preprocessing.
- [x] Implement sparse residual state with a scalar centering offset, cached
  base sum, and column summaries. Keep coordinate updates and cached-sum
  column dots at `O(nnz_j)`, and periodically refresh residuals to control
  drift.
- [x] Return original-scale coefficients and predictions while preserving any
  intercept induced by preprocessing, including through the typed
  Gaussian/L1 coordinate-descent API.
- [x] Add a documented KKT convergence check and report termination reason,
  iterations, and objective diagnostics. Distinguish iteration limits,
  numerical failures, and invalid input from convergence.
- [x] Implement a valid Gaussian lasso dual certificate and duality gap. Test
  feasibility, objective scaling, intercept and normalization policies, zero
  penalty, and zero reference loss before making relative duality gap the
  default convergence criterion.
- [x] Introduce `terminate_on(StoppingCriterion)` with criterion-specific
  tolerances, concise duality-gap and KKT constructors, and explicit
  absolute and relative gap tolerances. Document the reference scale and
  migrate the existing absolute-KKT `tolerance` setter without silently
  changing its meaning. Keep iteration budgets independent, reject
  unsupported criteria, and report the selected criterion, final value, and
  threshold.
- [x] Verify analytical cases, independent reference fixtures, agreement between
  dense and sparse fits, and agreement between explicit and lazy
  normalization. Test intercept policies, constant columns, penalty scaling,
  invalid inputs, and nonconvergence.
- [x] Add the benchmark harness and a lean consumer benchmark. Test sparse
  operation counts and reconstructed residuals, and measure time,
  allocations, and memory without forming a Gram matrix by default. Separate
  timing and heap harnesses cover dense and CSC fits; heap checks enforce
  linear memory budgets and release of fitted-result storage.

## 3. Composition proof

- [x] Extract small problem, datafit, penalty, result, and solver capabilities
  from the working lasso implementation. Keep matrix operations in
  LazyMatrix and solver state in Shrinkage.
- [x] Add proximal gradient, ridge (squared L2), and elastic net. Require a
  valid proximal map for the complete nonsmooth term.
- [x] Run the same iteration routine with concrete and object-safe oracles.
  Construct runtime problems by independently selecting the design operator,
  predictor datafit, and complete proximal term; keep dispatch outside
  scalar loops and generic column views inside concrete adapters.
- [x] Compare typed and runtime fits and return actionable errors for
  unsupported combinations.
- [x] Establish reusable workspace ownership across iterations and path points.
  Measure normalized-product and proximal allocations, including LazyMatrix
  0.5.0's scaled input clone. Verify reusable normalization scratch storage
  upstream before claiming allocation-free normalized iterations.
- [x] Prototype a separate fallible block-reader capability with bounded buffer
  reuse and explicit borrowing contracts before stabilizing solver
  interfaces.
- [x] Propagate backend errors with their source and operation context. Inject
  read failures during preprocessing and a later solver iteration, and
  verify that partially written outputs are neither consumed nor reported as
  success.

## 4. Datafit and coefficient layout proof

- [x] Add binomial regression through proximal gradient with stable loss and
  derivative evaluation, intercept handling, and justified step-size
  selection.
- [x] Check derivatives with finite differences and verify fits against
  independent reference cases, including extreme linear predictors.
  Predictor and coefficient checks cover both logit tails; typed and runtime
  fits match reproducible high-precision Newton and active-set fixtures.
- [x] Add multivariate Gaussian fitting with explicit feature, response, and
  penalty axes before stabilizing coefficient layouts.
- [x] Compare separable multivariate cases with independent response fits and
  provide a small runnable example.

## 5. SLOPE and regularization paths

- [x] Implement SLOPE's exact proximal map, validate nonnegative, nonincreasing
  weights, and test proximal optimality.
- [ ] **SLOPE cluster state:** Represent coefficient ordering, signs, and
  equal-magnitude clusters. Test zero coefficients, ties, permutations, and
  reconstruction of the coefficient vector.
- [ ] **SLOPE cluster updates:** Implement Gaussian cluster-coordinate sweeps
  using the cluster state, residual updates, and cluster merges. Verify each
  restricted update against a small independently solved subproblem.
- [ ] **SLOPE hybrid iteration:** Alternate cluster sweeps with full proximal
  steps that allow cluster splitting. Start with typed Gaussian/SLOPE fits,
  reusable workspaces, and an appropriate full optimality check. Test
  merges, splits, iteration limits, and agreement with proximal gradient.
- [ ] **SLOPE storage and normalization:** Extend the hybrid solver to dense and
  CSC input with lazy normalization. Check explicit/lazy agreement,
  intercept policies, and prediction-preserving back-transformation.
- [ ] **SLOPE solver comparison:** Benchmark hybrid and generic proximal fits on
  distinct, tied, and highly clustered solutions. Record correctness, time,
  allocations, and memory before recommending the hybrid solver.
- [x] Add path APIs with warm starts and reusable workspaces. Add lasso
  screening and working sets with full optimality checks at each solution.
- [x] Implement maximum-penalty calculations and path sequences for each
  supported family and penalty. Keep independent penalty strengths explicit
  when tuning multiple penalties.

## 6. Broader penalties and solvers

### Penalty composition

- [x] **Scaled penalties:** Add a validated nonnegative scaling wrapper for
  penalty values and supported complete proximal maps. Test zero scaling,
  invalid strengths, and equivalence with existing built-in strengths.
- [x] **Unpenalized masks:** Support explicit unpenalized coordinates for
  penalties with an implemented restricted proximal map. Validate masks,
  preserve excluded coordinates, and reject unsupported coupled penalties.
- [ ] **Row composition:** Compose scaled and masked penalties over feature rows
  through the existing coefficient layout and `AxisPenalty`. Verify row
  indexing, penalty scaling, and exclusion of intercepts.
- [ ] **Group definitions:** Add validated disjoint groups with explicit weights
  and a policy for uncovered coefficients. Reject duplicate, overlapping,
  empty, and out-of-range groups in the disjoint representation.
- [ ] **Disjoint group composition:** Apply a complete vector penalty to each
  validated disjoint group with reusable gather/scatter buffers. Check value
  and proximal agreement with independent group calculations.
- [ ] **Penalty sums:** Represent independently weighted sums for value
  evaluation and explicit splitting solvers. Require a separately
  implemented complete prox before exposing a sum to proximal gradient.

### Convex penalties

- [ ] **Group lasso:** Implement the weighted disjoint-group L2 penalty and
  exact block soft-thresholding prox. Verify proximal optimality, zero
  groups, and Gaussian fits against independent reference cases.
- [ ] **Sparse-group lasso:** Implement the complete prox for L1 plus disjoint
  group L2 penalties. Check limiting cases against lasso and group lasso,
  and verify proximal optimality.
- [ ] **Group SLOPE:** Implement the penalty and complete prox for disjoint
  groups under documented group-size and weight conventions. Check equal
  group norms, ordering, zero groups, and proximal optimality.
- [ ] **Overlapping groups:** After an explicit splitting solver is available,
  support one documented overlapping-group formulation. Test shared
  coefficients and compare small solutions with an independent reference.
- [ ] **Nuclear norm:** Implement matrix penalty evaluation and singular-value
  thresholding for one optional linear algebra backend. Verify layout,
  rank-deficient cases, and proximal optimality.
- [ ] **Coefficient bounds:** Add validated lower and upper bounds as an
  indicator penalty with an exact projection. Test feasibility, infinite
  endpoints, fixed coordinates, and small constrained fits.
- [ ] **Additional constraints:** Select one concrete constraint needed by a
  supported model, document its projection or splitting requirements, and
  add a separate implementation item with acceptance checks.

### Convex solvers

- [ ] **FISTA:** Add accelerated proximal gradient for the existing smooth
  datafits and complete convex proximal terms, reusing product workspaces
  and backtracking. Document assumptions, check convergence at the returned
  iterate, and compare fits and work counts with proximal gradient.
- [ ] **Three-operator splitting:** Add an oracle and iteration for a smooth
  term plus two separately proximable convex terms. Start with one Gaussian
  penalty-plus-bounds example, document step assumptions, and verify a
  justified fixed-point stopping criterion.
- [ ] **ATOS:** Add adaptive three-operator splitting using the validated
  three-operator formulation. Document adaptation assumptions, report step
  and line-search diagnostics, and compare solutions and work counts with
  fixed-step three-operator splitting.
- [ ] **ADMM splitting:** Define one Gaussian penalty split, its subproblem
  contracts, and primal and dual residual conventions. Implement the oracle
  with tests for its operators and error propagation.
- [ ] **ADMM quadratic subproblem:** Implement the Gaussian coefficient
  subproblem for that split using products and reusable solve storage.
  Verify solve residuals and failures without forming a Gram matrix by
  default.
- [ ] **ADMM iteration:** Combine the validated split and subproblem solver with
  dual updates and residual-based stopping. Compare Gaussian lasso solutions
  with existing solvers, and test iteration and inner-solve limits.

### Nonconvex penalties and methods

- [ ] **MCP penalty:** Implement MCP values and scalar coordinate subproblems
  with validated shape parameters. Test thresholds, boundary cases, and
  subproblem optimality against an independent scalar reference.
- [ ] **SCAD penalty:** Implement SCAD values and scalar coordinate subproblems
  with validated shape parameters. Test thresholds, boundary cases, and
  subproblem optimality against an independent scalar reference.
- [ ] **Gaussian MCP fitting:** Add coordinate descent using the MCP subproblem,
  explicit initialization, and a justified stationarity check. Test multiple
  starts and report local convergence separately from convex guarantees.
- [ ] **Gaussian SCAD fitting:** Add coordinate descent using the SCAD
  subproblem and the established nonconvex diagnostics. Verify stationarity
  and initialization sensitivity without requiring identical local
  solutions.
- [ ] **Group MCP penalty:** Implement disjoint-group MCP values and valid block
  subproblems. Verify block optimality on small independent reference cases
  and check reduction to scalar MCP for singleton groups.
- [ ] **Gaussian group MCP fitting:** Add block descent using the validated
  group MCP subproblem. Check final stationarity, initialization
  sensitivity, and agreement with scalar MCP fits for singleton groups.
- [ ] **Group SCAD penalty:** Implement disjoint-group SCAD values and valid
  block subproblems. Verify block optimality on small independent reference
  cases and check reduction to scalar SCAD for singleton groups.
- [ ] **Gaussian group SCAD fitting:** Add block descent using the validated
  group SCAD subproblem. Check final stationarity, initialization
  sensitivity, and agreement with scalar SCAD fits for singleton groups.
- [ ] **Majorization-minimization:** Implement one Gaussian MCP or SCAD
  formulation with explicit surrogate and inner-solve requirements. Verify
  surrogate validity, objective descent, and the final stationarity check.
- [ ] **Difference-of-convex fitting:** Implement one MCP or SCAD decomposition
  using an existing convex subsolver. Document inner tolerances and compare
  stationarity and initialization sensitivity with coordinate descent.
- [ ] **Specialized solver comparison:** Benchmark implemented block and
  nonconvex methods on documented supported combinations. Compare convex
  solutions where applicable and nonconvex stationarity otherwise; record
  evidence before adding solver-selection preferences.

## 7. Additional outcomes and statistical semantics

### Weights, offsets, and GLMs

- [ ] **Weight conventions:** Specify observation-weight validation, loss
  normalization, zero-weight behavior, and how weights affect fitted
  preprocessing and null models. Record the decision in `DESIGN.md`.
- [ ] **Weighted Gaussian fits:** Apply the documented weight conventions to
  Gaussian proximal fits. Test derivatives, zero weights, intercepts, and
  agreement with independently computed weighted least squares cases.
- [ ] **Weighted binomial fits:** Add observation weights to binomial proximal
  fits using the same conventions. Check derivatives, extreme predictors,
  and independent weighted reference fits.
- [ ] **Gaussian offsets:** Add validated predictor offsets to Gaussian proximal
  fits and prediction. Check dimensions, finite values, intercept
  interaction, and coefficient back-transformation.
- [ ] **Binomial offsets:** Extend the offset contract to binomial proximal fits
  and prediction. Check extreme predictors, intercept interaction, and
  independent reference fits.
- [ ] **Weighted and offset Gaussian paths:** Extend Gaussian null models,
  maximum penalties, and warm-start paths to weights and offsets. Compare
  with cold fits and test degenerate null models.
- [ ] **Weighted and offset binomial paths:** Extend binomial null models,
  maximum penalties, and warm-start paths to weights and offsets. Compare
  with cold fits and reject unsupported boundary null models.
- [ ] **Poisson datafit:** Implement stable Poisson loss and predictor
  derivatives with validated responses, weights, and offsets. Document loss
  normalization and test finite differences and overflow behavior.
- [ ] **Poisson fitting:** Fit Poisson models through proximal gradient with
  intercepts and original-scale prediction. Verify independent reference
  fits, all-zero responses, and numerical and iteration-limit diagnostics.
- [ ] **Poisson paths:** Implement the supported null model and maximum-penalty
  calculation, including weights and offsets. Compare warm paths with
  independent cold fits and reject nonfinite null optima.
- [ ] **Next GLM:** Select one additional family with a concrete use case.
  Specify its response, link, weight, offset, and intercept conventions, and
  add separate datafit and fitting tasks.

### Multinomial and multivariate models

- [ ] **Multinomial conventions:** Choose identifiability, class encoding,
  coefficient layout, intercept handling, and penalty axes. Document how
  these choices affect prediction and normalization.
- [ ] **Multinomial datafit:** Implement stable loss and predictor gradients
  under those conventions. Validate class inputs and check finite
  differences, extreme logits, and invariance properties.
- [ ] **Multinomial fitting:** Add proximal fitting with the chosen
  identifiability constraint and existing compatible penalties. Verify
  probabilities, back-transformation, and independent reference fits.
- [ ] **Multivariate model extension:** Select the next multivariate datafit and
  specify response coupling, coefficient layout, and penalty axes. Add
  separate datafit and solver-integration tasks for that model.
- [ ] **Multivariate group penalties:** Integrate the implemented group and
  sparse-group penalties with feature rows of multivariate Gaussian
  coefficients. Test axis semantics, intercept exclusion, and reference
  solutions with coupled responses.

### Survival models

- [ ] **Cox response and risk sets:** Validate event indicators and times,
  choose a tie convention, and implement reusable risk-set ordering. Test
  tied times, censored observations, and invalid inputs.
- [ ] **Cox datafit:** Implement partial likelihood and predictor gradients with
  the chosen tie convention. Keep risk-set coupling in the datafit, and
  verify derivatives and independent likelihood calculations.
- [ ] **Cox fitting:** Add proximal fitting with documented normalization,
  intercept, weight, and offset support. Verify stationarity and independent
  reference fits, and reject unsupported metadata explicitly.
- [ ] **Stratified Cox:** Extend risk sets, loss, and fitting to validated
  strata. Compare with sums of stratum-specific likelihoods and reference
  fits, including strata with no events.

### Curvature-based solvers

- [ ] **Curvature capability:** Derive a small fallible predictor-curvature or
  weighted-product capability from binomial fitting. Check derivatives and
  error propagation without adding requirements to unrelated datafits.
- [ ] **Binomial IRLS:** Use the curvature capability for ridge-regularized
  binomial fitting with stable weights and globalization. Compare with
  proximal gradient and test extreme predictors and failed inner solves.
- [ ] **Binomial proximal Newton:** Solve binomial/L1 quadratic subproblems with
  a compatible inner solver and line search. Report inner accuracy, verify
  final optimality, and compare with proximal-gradient reference fits.
- [ ] **Further specialized methods:** Choose one additional datafit/solver
  pairing only after its required curvature or product capability is
  justified. Add a bounded implementation task and reference checks.

## 8. Out-of-core execution

- [ ] **Reference file-backed adapter:** Promote the existing buffered Zarr
  demonstration into a documented reference adapter on the tested fallible
  block contract. Retain `O(n)` and `O(p)` solver state in RAM, keep storage
  dependencies optional, and compare fits with in-memory input.
- [ ] **Storage access description:** Describe efficient row and column scans,
  block geometry, and costly access for the reference adapter. Verify the
  description against instrumented reads rather than inferring it from
  random-access support.
- [ ] **Streaming Gaussian kernels:** Add blockwise Gaussian loss and gradient
  accumulation with reusable buffers. Compare with full-vector evaluation,
  including normalization, intercepts, and injected read failures.
- [ ] **Streaming binomial kernels:** Add stable blockwise binomial loss and
  gradient accumulation using the streaming contract. Verify extreme
  predictors and agreement with in-memory fits.
- [ ] **Memory and block policy:** Add validated memory and block-size settings
  to the buffered consumer. Account for known buffers and document which
  limits are hard or advisory, including backend decoding storage.
- [ ] **Thread and cache policy:** Extend execution policy with thread count,
  cache capacity, and explicit permission for temporary copies, transposes,
  conversions, and disk caches. Test that prohibited operations are rejected
  before execution.
- [ ] **I/O diagnostics:** Instrument data passes, block reads, and bytes where
  measurable. Distinguish backend-reported I/O from estimates, and verify
  counters on deterministic small fits.
- [ ] **File-backed working sets:** Adapt Gaussian lasso working-set fitting to
  fallible buffered column access with policy-bounded storage. Reuse the
  existing numerical updates, verify read failures, and compare with
  in-memory fits before adding a cache.
- [ ] **Active-column cache:** Add a policy-bounded cache for a supported
  file-backed coordinate or working-set method. Verify eviction, cached and
  uncached agreement, and periodic full optimality scans.
- [ ] **File-backed workspace contract:** Choose one `O(n)` workspace to move
  out of RAM and define its fallible block access, lifetime, and temporary
  storage requirements. Keep coefficients and essential `O(p)` state in RAM.
- [ ] **File-backed workspace implementation:** Implement that workspace for one
  Gaussian solver using the agreed contract. Compare with the RAM workspace
  and inject failures during updates and final checks.
- [ ] **Out-of-core memory checks:** Add allocation and buffer checks for the
  reference adapter and workspace implementation. Detect hidden full-matrix
  materialization and verify behavior when the configured budget is too
  small.
- [ ] **Out-of-core benchmarks:** Compare in-memory and file-backed fits over
  representative block sizes and cache policies. Report passes, I/O, peak
  memory, and time alongside numerical agreement.

## 9. Bindings, selection, and model evaluation

### Runtime specifications and selection

- [ ] **Runtime fit specifications:** Extend the existing runtime API with
  validated family, penalty, solver, and normalization specifications for
  implemented combinations. Test typed/runtime agreement and actionable
  compatibility errors.
- [ ] **Runtime path specifications:** Expose explicit and automatic path
  specifications through the runtime facade. Preserve independent penalty
  strengths and compare ordering, warm starts, and failures with typed
  paths.
- [ ] **Runtime prediction:** Expose fitted linear predictors and supported
  response transformations through the runtime facade. Verify original-scale
  coefficients, preprocessing, offsets, and errors on incompatible inputs.
- [ ] **Compatibility matrix:** Publish a tested matrix of supported families,
  penalties, solvers, layouts, and storage capabilities. Link unsupported
  selections to useful alternatives.
- [ ] **Automatic solver selection:** Implement a documented deterministic
  `auto` policy using the compatibility matrix, efficient storage access,
  memory policy, and measured solver preferences. Report the selected solver
  and test explicit overrides and unsupported requests.

### Cross-validation

- [ ] **Cross-validation folds and metrics:** Add validated deterministic fold
  assignments and family-appropriate scoring for Gaussian and binomial
  models. Test coverage, invalid folds, and score normalization.
- [ ] **Cross-validation paths:** Fit preprocessing on each training fold and
  reuse it across that fold's path. Verify held-out predictions and prove
  absence of preprocessing leakage with a targeted fixture.
- [ ] **Cross-validation selection:** Aggregate fold scores, select a tuning
  point under a documented rule, and refit on all training observations.
  Test reproducibility, failed fits, and independently checked selections.

### Python bindings

- [ ] **Python input contract:** Add the minimal optional binding package and
  validate dense array ownership, lifetime, layout, and mutability. Test
  accepted zero-copy inputs and explicit copies or errors for other inputs.
- [ ] **Python fitting and prediction:** Expose Gaussian lasso through the
  runtime facade with fitted diagnostics, prediction, and translated errors.
  Compare with Rust results and keep callbacks outside numerical loops.
- [ ] **Python paths:** Expose explicit and automatic regularization paths with
  validated output shapes and independent penalty strengths. Compare
  predictions, diagnostics, and failures with Rust paths.
- [ ] **Python cross-validation:** Bind the Rust cross-validation API and verify
  selected penalties, refits, and leakage checks against Rust results.

### R bindings

- [ ] **R input contract:** Add the minimal optional binding package and
  validate matrix ownership, lifetime, layout, and mutability. Test accepted
  zero-copy inputs and explicit copies or errors for other inputs.
- [ ] **R fitting and prediction:** Expose Gaussian lasso through the runtime
  facade with fitted diagnostics, prediction, and translated errors. Compare
  with Rust results and keep callbacks outside numerical loops.
- [ ] **R paths:** Expose explicit and automatic regularization paths with
  validated output shapes and independent penalty strengths. Compare
  predictions, diagnostics, and failures with Rust paths.
- [ ] **R cross-validation:** Bind the Rust cross-validation API and verify
  selected penalties, refits, and leakage checks against Rust results.

### Coverage and documentation

- [ ] **Binding coverage plan:** After the first fit/path/predict interfaces
  work, list remaining implemented families and input backends for each
  language. Add one bounded integration task per family or backend, with
  Rust parity and ownership checks.
- [ ] **Statistical API documentation:** Document supported objectives,
  normalization, intercepts, weights, offsets, stopping tolerances, and
  diagnostics for the public runtime and binding interfaces.
- [ ] **Rust user examples:** Add runnable runtime fit, path, prediction, and
  cross-validation examples. Verify outputs against shared reference cases.
- [ ] **Python user examples:** Add runnable binding examples for fitting,
  paths, prediction, and cross-validation. Verify agreement with the Rust
  examples and document input ownership and copies.
- [ ] **R user examples:** Add runnable binding examples for fitting, paths,
  prediction, and cross-validation. Verify agreement with the Rust examples
  and document input ownership and copies.
- [ ] **Performance documentation:** Publish measured performance for supported
  typed, runtime, and binding combinations, including copies, allocations,
  and memory. Keep advertised coverage within the tested matrix.

## 10. Optional accelerators

- [ ] **Accelerator workload:** Select one representative bulk workload and
  optional device backend after the CPU interfaces are proven. Record CPU
  baselines and define numerical and performance acceptance checks.
- [ ] **Device storage:** Add optional device buffers and explicit upload and
  download operations for that backend. Validate shape, ownership, and
  lifetime contracts without requiring host slices in universal interfaces.
- [ ] **Device products:** Implement forward and transpose products for the
  selected workload. Compare with CPU kernels and test backend failures and
  transfer accounting.
- [ ] **Device datafit:** Implement loss and gradient kernels for the selected
  datafit, including reductions and intercept handling. Verify CPU agreement
  across ordinary and extreme inputs.
- [ ] **Device proximal map:** Implement the workload's complete proximal term
  with reusable device scratch storage. Check proximal optimality and CPU
  agreement.
- [ ] **Device solver:** Connect the validated kernels through a device oracle
  and iteration routine. Keep matrices, coefficients, gradients, and scratch
  storage on the device between iterations, and report transfer diagnostics.
- [ ] **Device fit verification:** Compare complete CPU and device fits,
  stopping diagnostics, and original-scale predictions. Test iteration
  limits, backend errors, and the lean build without GPU dependencies.
- [ ] **Accelerator benchmarks:** Measure transfers, device memory, kernel work,
  and end-to-end time against the CPU baseline. Document which workload
  sizes benefit before recommending device execution.

## Deferred work

- [ ] **Original-scale L1:** Derive and implement the transformed weighted L1
  subproblem under nontrivial normalization. Verify equivalent predictions
  and objectives against an explicitly transformed reference problem.
- [ ] **Other original-scale penalties:** Choose one additional penalty only
  when a consumer needs it. Derive its transformed subproblem and add a
  separate implementation task; reject unsupported original-scale requests.
- [ ] **Workspace split assessment:** Use actual dependency-isolation or reuse
  requirements to decide whether one concrete crate boundary is justified.
  Record the decision and add a migration task only if needed.
- [ ] **Selection planner assessment:** Use measured failures or costs of the
  deterministic `auto` policy to identify one needed improvement. Specify
  its inputs, benchmark, and a separate implementation task.

Distributed execution, general symbolic optimization, out-of-core coefficient
vectors, and global optimization of arbitrary nonconvex objectives remain
outside the initial scope.
