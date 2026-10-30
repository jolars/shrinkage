# Profiling Shrinkage

Use samples or phase timers to attribute cost, then ordinary uninstrumented
builds to establish a speedup. The development shell supplies
`cargo-flamegraph`, `samply`, `hyperfine`, and LLVM inspection tools. Linux also
supplies `perf`, Valgrind, and `taskset`. DHAT allocation harnesses already live
in this repository. These tools do not grant kernel profiling permissions.

## Follow callers into the responsible layer

  | Layer                                                                 | Source                                                              |
  | --------------------------------------------------------------------- | ------------------------------------------------------------------- |
  | Python configuration and fitted result conversion                     | `packages/python/src/shrinkage/`                                    |
  | Owned NumPy copies, finite-value validation, and detached native fit  | `packages/python/rust/input.rs` and `packages/python/rust/lasso.rs` |
  | Statistical validation and normalization preparation                  | `src/problem.rs` and its child modules                              |
  | Gaussian coordinate updates, residual state, KKT and dual diagnostics | `src/solver/coordinate.rs`                                          |
  | Working-set membership and scores                                     | `src/solver/working_set.rs`                                         |
  | Original-loss IRLS checks and weighted subproblems                    | `src/solver/irls.rs`                                                |
  | Proximal products and vector adapters                                 | `src/solver/proximal.rs` and `src/solver/proximal/`                 |
  | Hybrid transitions, aggregate columns, and cluster state              | `src/solver/hybrid.rs` and `src/solver/hybrid/`                     |
  | Penalty evaluation and proximal operators                             | `src/penalty.rs` and `src/penalty/`                                 |
  | Path preparation and workspace reuse                                  | `src/path.rs` and `src/path/`                                       |

Follow matrix calls into the resolved published LazyMatrix source when needed.
Compare matrix passes and arithmetic before suggesting a backend replacement.
Normalization algebra belongs to LazyMatrix; statistical validation, residual
state, and stopping decisions belong to Shrinkage. A sibling checkout can have
unpublished APIs and is not the implementation selected by the lockfile.

## Sample an optimized build with symbols

Keep profiling outputs separate from normal release and benchmark artifacts. Use
Cargo profile environment overrides rather than modifying optimization settings
or the library's MSRV. For a focused Criterion capture:

```sh
mkdir -p .build/perf-investigation
compute bench --reason "Profile dense lasso" -- devenv shell -- bash -c '
  export CARGO_TARGET_DIR="$PWD/.build/perf-investigation/target"
  export CARGO_PROFILE_BENCH_DEBUG=2 CARGO_PROFILE_BENCH_STRIP=none
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C force-frame-pointers=yes"
  cargo flamegraph --profile bench --no-default-features --features faer_v0_24 \
    --bench lasso --output .build/perf-investigation/lasso.svg \
    --cmd "record -e cycles:u -F 997 --call-graph fp --user-callchains -o .build/perf-investigation/lasso.data" \
    -- --bench lasso_1024x64/dense --exact --profile-time 10
'
```

Choose a fresh output directory for each investigation or capture. Adapt the
benchmark, feature set, filter, and duration. Check the local tool help when
changing options and verify that the filter runs the intended case. Criterion's
profile mode repeats work without collecting ordinary timing estimates; samples
can include setup that the benchmark timer excludes.

For a prebuilt repeated-fit probe, record directly with:

```sh
perf record -e cycles:u -F 997 --call-graph fp --user-callchains \
  -o .build/perf-investigation/probe.data -- <executable> <arguments>
```

Wrap the recording in `compute bench` as above. Get Cargo executable paths from
JSON build artifacts rather than a wildcard that may select stale binaries or
`.d` files. A focused production-code probe can separate setup from repeated
solves without exporting new public APIs or reimplementing the hot kernel.

Use both `cycles:u` and `--user-callchains`. Excluding kernel events alone can
still leave a request for kernel callchains, which can fail under
`perf_event_paranoid=2`. Retry with explicitly user-only callchains before
concluding that perf is unavailable. Do not automatically change host-wide
sysctls or request root access. Check the recorded event on hybrid CPUs and
report whether it measures cycles or a fallback event.

Inspect inclusive and self costs separately:

```sh
perf report --stdio --children --inline -i .build/perf-investigation/lasso.data
perf report --stdio --no-children --no-inline -i .build/perf-investigation/lasso.data
```

Check stack depth, symbols, inline frames, lost samples, and the loaded shared
library before interpreting percentages. A DSO filter changes the denominator
and can hide Python, NumPy, and allocator work. Inclusive shares overlap. Use
`llvm-objdump` or `llvm-nm` when inlining or vectorization needs inspection.

## Python builds and fallbacks

For a diagnostic extension, build the separate `packages/python` workspace with
release debug information, no stripping, frame pointers, and a separate target
directory. Apply release profile overrides there rather than assuming root
profile settings carry across workspaces. Install or explicitly load that build
in an isolated environment and record its path and hash. A normal editable
release installation should remain available for final timing.

Set numerical thread limits before starting Python, warm imports and the
selected workload, and repeat verified fits long enough to sample native work.
Check the profile includes the intended Rust functions. Python startup, dataset
generation, independent evaluators, and protocol work can dominate a short
whole-process capture.
`samply record --save-only --output <profile.json.gz> <command> <args>` is an
alternative when its local permissions and unwinder work.

If sampling remains unavailable, use scoped phase timers or deterministic work
counters and disclose the limitation. Time input preparation, workspace setup,
updates, residual reconstruction, and KKT and dual passes separately where
appropriate. Compare paired instrumented and plain results, work counts, and
wall times to assess overhead. Prefer an observer or an isolated diagnostic
build to permanent timers in scalar loops. Remove temporary instrumentation and
verify the normal extension is restored before final measurements.

For allocation questions, start with the existing DHAT harnesses. Use Valgrind
Callgrind or Massif only for a focused native probe when their instrumentation
answers the question. Their elapsed times are not performance estimates.
`hyperfine` measures command startup and teardown too; use it for suitably long
prebuilt commands, not as a replacement for fit-only timers. Use `taskset` when
CPU placement matters, recording affinity rather than silently adding it to one
contestant.
