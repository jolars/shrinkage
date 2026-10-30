---
name: shrinkage-perf-investigation
description: Investigate performance in Shrinkage using Rust solver and kernel
  benchmarks, Python comparisons, allocation accounting, and profiling. Use for
  solver regressions, matrix or normalization hotspots, convergence-check costs,
  Python binding overhead, and measured optimization work.
---

# Investigate Shrinkage Performance

Reproduce the requested workload and explain its cost with measurements. An
investigation can finish with a supported diagnosis and ranked next steps. When
the user requests an optimization, carry supported changes through numerical
verification and fresh uninstrumented timing.

This skill belongs to the Shrinkage repository. Run from its root, read
`AGENTS.md`, and inspect the relevant harness and implementation. Consult
`DESIGN.md` for numerical conventions. Use `devenv shell` for the pinned library
toolchain. The Python extension has a separate Cargo workspace.

## Establish comparable work

Keep the user's solver, workload, and competitor. If no case is specified,
choose a bounded existing case and state the choice. Match data, matrix layout,
precision, normalization, intercept treatment, penalty, initialization, thread
counts, and timing boundaries. Record resolved backend versions and features. A
dense Fortran-contiguous array and a strided view can exercise different work.

Distinguish two questions:

- **Solver effectiveness:** time to a common independently checked accuracy
  target, or quality within the same budget. Retain failures and budget misses.
- **Implementation cost:** identical kernel inputs or matched numerical work.
  Compare feature visits, full checks, inner iterations, matrix products, and
  allocations where relevant. Equal iteration limits do not imply equal work.

Check the actual objective and termination semantics before timing. Preserve
`P(beta_0, beta) = F(beta_0, beta) + R(beta)`, the sample normalization of the
loss, and the complete penalty. Equal tolerance values across packages do not
establish equal accuracy. For lasso, independently evaluate the primal
objective, a feasible dual bound, and KKT violation outside the fit timer. For
other models, use their applicable stationarity or optimality certificate. A
budget-limited fit may supply an intermediate convergence point but does not
certify success. Compare predictions and objective values when coefficients need
not be unique.

Check whether each iteration is a complete working-set sweep, an outer IRLS
step, a proximal step, or a hybrid transition containing multiple inner sweeps.
Identify check frequency, residual refreshes, screening, and working-set growth.
A diagnostic matrix product or prox call is additional work, not an accepted
solver step. Retain algorithm differences as part of the practical competitor
comparison and isolate matched kernels when explaining implementation overhead.

## Measure and locate the cause

Read [Benchmarking](references/benchmarking.md) to select a harness, preserve
baseline builds, and interpret Python timers and allocation statistics.
Establish an uninstrumented baseline before production edits. Keep baseline and
candidate measurements in one `compute bench` reservation. Wrap builds and
correctness checks in `compute run` outside that reservation; do not nest
reservations. Keep commands in the foreground and let queued work wait.

Read [Profiling](references/profiling.md) for module ownership, optimized
sampling builds, user-space perf options, and fallbacks. Measure final speedups
with ordinary timing builds, separately from sampling, phase timers, or heap
instrumentation. Follow inclusive caller costs before interpreting hot leaves.

Separate these possible causes with a focused experiment:

- Setup, repeated finite-value scans, normalization statistics, and workspace
  construction versus iterative numerical work.
- Python validation and owned input copies versus the detached Rust solver,
  output conversion, and benchmark protocol work.
- Coordinate or proximal updates versus restricted checks, full KKT and dual
  correlations, residual reconstruction, and working-set selection.
- Matrix layout, scalar reductions, normalization corrections, vector copies,
  dispatch, and scratch allocation versus the statistical algorithm itself.
- For IRLS and hybrid solvers, accepted outer work versus weighted subproblems,
  rejected steps, aggregate construction, and cluster maintenance.

Record deterministic work counts where possible. For example, count and time
full matrix passes before concluding that coordinate updates are slow. Report
the denominator for phase shares. Nested inclusive costs overlap, and a smaller
profile percentage alone does not establish less elapsed time.

## Verify a change and retain evidence

Test one supported hypothesis at a time. Changes to check cadence, tolerances,
screening, precision, or solver algorithms change numerical work and must be
identified separately from arithmetic-preserving kernel improvements. Preserve
fresh final diagnostics, iteration-limit semantics, sparse residual algebra,
penalty scaling, and prediction-preserving back-transforms. Verify that excluded
features can reenter and that paths reset or reuse state correctly when
affected.

Add a failing regression test before fixing observable behavior. Prefer stable
work counters and numerical result checks to wall-time assertions. Rebuild the
Python extension before Python correctness checks and verify which library is
loaded. Run the scope-appropriate validation in `AGENTS.md`, then remeasure the
original workload and representative contrasting shapes, sparsities, penalty
strengths, or correlations when claiming a general improvement.

Keep profiles, raw measurements, source diffs, and separate native artifacts in
fresh ignored `.build/` or `target/` directories. Record commands, source and
library hashes, toolchain and package versions, features, seeds and data hashes,
hardware, thread settings, attained accuracy, and uncertainty. Use isolated
checkouts for historical sources rather than swapping production files or
installed extensions in the user's working tree. Refresh tracked benchmark
figures when requested; an investigation alone need not publish new results.

Report measured findings separately from hypotheses: the workload and reference,
timing scope, medians and variation, work and accuracy differences, responsible
functions, verified improvement, validation, tooling limitations, artifact
paths, and the next supported experiment. Treat differences within timing
variation as inconclusive. Remove temporary instrumentation and unsupported
experimental edits without disturbing user changes.
