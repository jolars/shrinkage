# Benchmarking Shrinkage

Read the selected source and manifest before running it. Commands below run from
the repository root. Use the smallest harness that answers the question.

## Select an existing harness

  | Question                                                       | Starting point                                                                                                           |
  | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
  | Complete dense and CSC Gaussian lasso fits                     | `benches/lasso.rs`                                                                                                       |
  | Python lasso versus scikit-learn at common accuracy            | `packages/python/benchmarks.yaml`, `packages/python/benchmarks/lasso.py`, and `packages/python/benchmarks/report.py`     |
  | Working-set work and screening costs                           | `benches/working_sets.rs` and `benches/screening.rs`                                                                     |
  | Weighted coordinate IRLS versus proximal fitting               | `benches/irls.rs` and `benches/hybrid_irls.rs`                                                                           |
  | Warm paths versus independent cold fits                        | `benches/hybrid_paths.rs` and path tests                                                                                 |
  | Native products, normalized products, and vector bridge copies | `benches/proximal.rs` and `benches/proximal_buffers.rs`                                                                  |
  | Complete clustered, chain-fused, or SLOPE hybrid fits          | `benches/clustered_lasso.rs`, `benches/chain_fused_lasso.rs`, and `benches/slope_hybrid.rs`                              |
  | Proximal maps and penalty composition                          | `benches/slope.rs`, `benches/group_lasso.rs`, `benches/scaled.rs`, `benches/masked.rs`, and `benches/row_composition.rs` |
  | Fit allocations, reusable buffers, and peak live heap          | The corresponding `benches/*_heap.rs` harness                                                                            |

`Taskfile.yml` provides `bench:*` tasks, including buffer, path, and heap
checks. Not every executable in `benches/` uses Criterion. Inspect `Cargo.toml`
for required features and the source for filtering support. Prefer versioned
backend features for recorded measurements. Do not use all features for timing
unless that exact configuration is the workload under investigation.

For a focused lasso run:

```sh
compute bench --reason "Dense lasso baseline" -- devenv shell -- \
  cargo bench --locked --no-default-features --features faer_v0_24 \
  --bench lasso -- 'lasso_1024x64/dense' --exact
```

The lasso harness constructs data before timing, then measures complete fitting,
including Rust input validation and preprocessing. It verifies convergence
before timing. Criterion filters, batching, and black-box inputs do not
establish that competing harnesses charge equivalent setup or teardown.

Use separate checkouts and target directories for historical builds. Keep raw
Criterion estimates and exact build identities. Prebuild contestants and
alternate measurement order when drift could obscure a small effect. Keep the
whole paired comparison in one `compute bench` reservation. Use `compute run`
for preparation outside it; nested reservations fail. Avoid measuring debug
builds, tracing observers, or a copied implementation of the suspected
production kernel.

## Python comparison

The public reproduction command rebuilds the release extension, checks the
harness, measures both packages, and overwrites the tracked JSON and figure:

```sh
compute bench --reason "Python lasso versus scikit-learn" -- devenv shell -- \
  task python:bench
```

Use it when refreshing the published comparison is in scope. For diagnosis
alone, use the same manifest with the Metewand CLI's `check`,
`run --output-dir`, and `export --format json --require-complete` commands,
writing to a fresh ignored `.build/` directory. See
`packages/python/benchmarks/README.md` for CLI setup and `report.py` for the
exact invocation. The CLI is installed separately from the locked Python worker
package; verify compatibility rather than assuming a sibling checkout or an
executable on PATH. `METEWAND` selects an explicit executable.

For ordinary package preparation:

```sh
compute run -- devenv shell -- uv sync --locked --project packages/python \
  --group benchmark --reinstall-package shrinkage \
  --config-settings-package 'shrinkage:build-args=--profile release'
```

`cargo build` alone does not install the Python extension. Record the loaded
`shrinkage._shrinkage.__file__` and its SHA-256 hash. Historical checkouts need
separate virtual environments and native artifacts as well as separate Cargo
outputs; otherwise an editable rebuild can silently replace the contestant.
After preparation, `uv run --no-sync --project packages/python ...` keeps
synchronization out of measurements.

The current lasso worker sets numerical thread limits before importing NumPy.
Verify actual threadpool settings and retain them when adapting a probe. Both
methods receive the same centered Fortran-contiguous float64 arrays, disable
intercepts, and start from zero. Shrinkage disables column standardization. The
independent evaluator recomputes objective, feasible dual bound, and KKT
violation. The target is relative gap `1e-8` against the null-model half-squared
loss. Scikit-learn uses `tol=5e-9` because its gap threshold uses twice that
reference loss; inspect the installed version's actual stopping rules too.

The plotted `implementation_time_ns` includes model construction, API
validation, input copies, and fitting. Loading arrays, coefficient extraction,
independent accuracy checks, serialization, and statistics lie outside it. The
exported `execute_only` wall time has a wider boundary, including result
conversion and protocol work. Do not substitute one timer for the other.

Each `fresh_sequence` observation is a separately timed fit from zero to its own
iteration limit. Times are neither incremental sweep costs nor samples from one
continuous fit. Profiles stop on independently verified accuracy. Compare
medians at the common target, retain intermediate gaps and failure history, and
check that the schedule is long enough after any change to working-set or check
cadence. The current Gaussian CD checks initially, every ten working-set sweeps,
and at the iteration limit; intermediate truncated fits therefore perform a
final check that a longer unconstrained fit may defer. Read the implementation
and manifest instead of applying this schedule to every solver.

## Allocation and numerical checks

Heap harnesses use DHAT separately from timing. For example:

```sh
compute run -- devenv shell -- cargo bench --locked --no-default-features \
  --features faer_v0_24 --bench lasso_heap
```

The lasso heap harness creates input data before measurement, counts fitting and
its returned result, and checks that dropping the result releases retained
storage. `total_blocks` counts allocations, `total_bytes` is cumulative
requested memory, and `max_bytes` is peak live instrumented heap. None is
process peak RSS. Use the other heap harnesses for their respective solver and
workspace scopes.

Use independent numerical checks outside timers and source-level work counters
inside focused correctness tests. `WorkingSetDiagnostics` distinguishes feature
visits, full checks, restricted checks, and membership changes. Interpret these
in their solver context: a full IRLS check is not interchangeable with a
Gaussian CD check. Hybrid fits also distinguish accepted PG steps and coordinate
sweeps. Avoid adding callbacks to both solvers merely to obtain convergence
plots when independent truncated fits already answer the question.
