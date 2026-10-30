# Python lasso benchmark

This comparison plots lasso convergence for shrinkage and scikit-learn on one
dense Gaussian problem at one regularization strength. The horizontal axis uses
linear fit time, and the logarithmic vertical axis shows the independently
computed duality gap. Metewand runs the methods, evaluates their answers, and
retains the measurements. The authored Python documentation uses a plot
generated from a JSON snapshot of the completed run.

Install the Metewand CLI separately, using a revision that supports
`export --format json --require-complete`. The Python dependency group pins the
worker package independently; both use binding API version 1:

```sh
compute run -- rustup toolchain install 1.98.0 --profile minimal
compute run -- rustup run 1.98.0 cargo install --locked --git https://github.com/jolars/metewand \
  metewand
```

The CLI needs its own Rust 1.98 toolchain when built from source. Keep
shrinkage's pinned toolchain for the Python extension. An existing compatible
CLI also works: set `METEWAND=/absolute/path/to/metewand` to select it.

From the shrinkage repository root:

```sh
compute bench --reason "Python lasso versus scikit-learn" -- devenv shell -- task python:bench
```

The task rebuilds shrinkage in release mode, runs worker and analytic reference
checks, measures both implementations within the same compute reservation, and
exports the results. Each run gets a new directory under `.build/python-lasso/`.
Keep that directory for the full Metewand results, provenance, and exports. The
documentation snapshot retains the dataset seed and hash, plus every measured
convergence point, including both the fit timer and the executor's wall timer.
Metewand rejects incomplete exports before writing the snapshot. Successful
retries can supply the selected measurements; full attempt history remains in
the retained run directory. The reporting code checks the assumptions specific
to this figure: shared data, one-thread execution, fit timers, and enough
measurements for its error bars.

The saved `docs/figures/lasso.json` is Metewand's `results.json` analysis
export. The report copies it directly, then plots selected observations with
complete metric dictionaries. Repetition counts and observation settings come
from that export. Package versions and the native-extension hash are recorded by
the workers outside the fit timer.

The `fresh_sequence` profile runs five repetitions per method, with increasing
iteration limits from 1 through 12 sweeps. Each point starts a new model from
zero coefficients. An independent evaluator computes both the absolute duality
gap for plotting and its ratio to the null-model loss for stopping. Metewand
stops each profile when that ratio reaches `1e-8`. Early points with larger gaps
are valid observations. The reporting tests check that these points survive
export processing and that each plotted time belongs to its own fit.

Each marker shows the median time and gap at one iteration limit. Horizontal
error bars span the middle half of five timings. Lines connect the limits in
increasing order. These are separately timed fits from zero initialization.

The figure uses Metewand's in-worker timer around model construction and
`.fit()`. This includes input validation, copies required by each API, and
numerical fitting. Array loading, imports, independent evaluation, coefficient
extraction, and result serialization happen outside that timer. Metewand also
retains its primary `execute_only` wall time, which includes result conversion
and protocol work. Worker reuse avoids repeated imports.

Redraw or check the saved figure without measuring anything:

```sh
uv run --locked --project packages/python --group benchmark python packages/python/benchmarks/report.py
uv run --locked --project packages/python --group benchmark python packages/python/benchmarks/report.py --check
```

`task python:bench:check` checks the reporting code and the saved figure without
taking new measurements.

Benchmark dependencies are optional development tools. Ordinary package installs
and documentation builds do not require them, the Metewand CLI, or a benchmark
run.
