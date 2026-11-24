# Gaussian lasso in R

Use the repository development environment with R 4.4 or later, or provide the
same dependencies on your PATH. Install glmnet 5.1 or later, Matrix, digest, and
jsonlite in the R environment used to prepare the workspace. Install Metewand
for R from the `metewand-r-v0.4.0` tag of
[jolars/metewand](https://github.com/jolars/metewand) and put the separate
Metewand 0.9 CLI on PATH. Set `METEWAND` to select an explicit CLI executable.
For example, install the worker package with pak:

```r
pak::pak("jolars/metewand/packages/r@metewand-r-v0.4.0")
```

```sh
task r:bench:prepare -- .build/r-glmnet/current
task r:bench -- .build/r-glmnet/current smoke .build/r-glmnet/smoke
task r:bench -- .build/r-glmnet/current compare .build/r-glmnet/compare --dataset sparse
```

Preparation compiles the checkout's Rust core and the R binding in release mode
and stages the worker libraries. Prepare a new workspace after changing code or
dependencies. Reuse a prepared workspace to rerun the same build. Use
`published` as the second preparation argument to build against the published
core declared by the R package.

The `smoke` experiment checks small dense and sparse problems. The `compare`
experiment measures five fits per method and dataset at the common accuracy
target. Change the staged manifest's dataset parameters or add Metewand named
cases to compare other dimensions, seeds, densities, or lambda fractions.

```sh
metewand check --workers --manifest .build/r-glmnet/current/metewand.yaml
metewand --output json plan compare --manifest .build/r-glmnet/current/metewand.yaml
metewand --output json status --output-dir .build/r-glmnet/compare/run
```

The runner writes Metewand JSON to stdout and prints the path to an analysis
export on stderr. Keep the run directory for provenance and diagnostics. Pass an
observation or attempt ID to `metewand explain ID --output-dir PATH` to inspect
its evidence. Repeat the original command to resume, and add `--retry` after
diagnosing a failed attempt. Use a new output directory for an independent
measurement.

For the vignette, prepare a workspace with `published` and run the `vignette`
experiment. Review its results before calling
`task r:bench:publish -- PATH/TO/results.json` to refresh the tracked table. The
vignette uses the R callback time. The export also retains the outer wall time,
which includes result conversion and protocol overhead.
