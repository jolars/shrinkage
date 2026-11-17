# Elastic net benchmark

This benchmark compares the time Shrinkage's Gaussian `Irls` solver and Linfa's
coordinate descent take to fit the same elastic net model at different levels of
accuracy.

![Rust elastic net fits for Shrinkage and Linfa. The horizontal axis shows fit
time in milliseconds. The logarithmic vertical axis shows maximum KKT violation,
with a dashed line at the shared accuracy target.](figures/elastic-net.png)

Each marker summarizes five independent fits at one solver tolerance. It shows
the median fit time and median maximum KKT violation. Horizontal bars span the
middle half of the measured times. Lines connect the points in order of
tightening tolerance. Each curve ends at the first fit in the schedule that
reaches the shared accuracy target of `1e-7`.

## Problem and accuracy

The dataset has 100 observations and 10,000 features in correlated groups of
eight. The generator uses seed `20261026` and forms the response from 10
features spaced evenly across the design, with coefficients alternating between
`+1` and `-1` and additive uniform noise. Both solvers receive the same
column-major `f64` matrix and response. They start at zero coefficients with no
intercept fitting or normalization. Shrinkage uses `ndarray` 0.17 and Linfa uses
`ndarray` 0.16. Both run on a single thread with pure Rust numerical
dependencies.

Both fits minimize the following objective with $\lambda = 0.125$ and
$\alpha = 0.5$.

$$
  \frac{\lVert y - X\beta\rVert_2^2}{2n}
  + \lambda\left( \alpha\lVert\beta\rVert_1 + \frac{1 - \alpha}{2}\lVert\beta\rVert_2^2 \right).
$$

Metewand requests progressively tighter solver tolerances from `1e-1` through
`1e-9`. Shrinkage checks an absolute KKT tolerance. Linfa checks relative
coefficient updates and a duality-gap threshold. A separate Rust evaluator
recomputes the objective and maximum coordinate KKT violation from the data and
returned coefficients. That violation determines when each profile has reached
the target. [Linfa stopping
rules](https://docs.rs/crate/linfa-elasticnet/0.8.1/source/src/algorithm.rs)

## Timing and reproduction

Metewand runs the measurements serially, randomizes solver order, and discards
one warm-up per worker. The worker times model construction and the public fit
call in release mode. Linfa computes fitted values inside `fit` and skips
coefficient variance estimation when the design has more features than
observations. Dataset loading, coefficient extraction, result serialization, and
independent evaluation happen outside the fit timer.

The [saved measurements](figures/elastic-net.json) record the hardware, compiler
and library versions, dataset and worker hashes, and each plotted fit's timing
and accuracy. The figure describes one synthetic dataset at one regularization
strength on the recorded machine.

From the Shrinkage repository root, build the sibling Metewand checkout and run
the comparison.

```sh
(cd ../metewand && devenv shell -- cargo build --locked --release --bin metewand)
devenv shell -- task bench:elastic-net
```

Set `METEWAND` to use a compatible CLI such as Metewand 0.8.0 from another
location. The native worker uses Unix protocol pipes. The task prints the
directory under `.build/rust-elastic-net/` that contains the full Metewand run,
verified JSON export, and local plot.

After changing the root Rust package version, refresh the benchmark's lockfile
with
`cargo update --manifest-path packages/rust-benchmarks/Cargo.toml -p shrinkage`.

To refresh the docs figure, pass the directory printed by the task to the report
script. Run the second command to check the saved figure without taking new
measurements.

```sh
uv run --script scripts/rust_elastic_net_benchmark.py --record .build/rust-elastic-net/run-ABC123
uv run --script scripts/rust_elastic_net_benchmark.py --check
```
