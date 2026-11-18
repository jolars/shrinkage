# Elastic net benchmark

This benchmark compares elastic net fits from Shrinkage's Gaussian `Irls`
solver, Linfa's coordinate descent, and SmartCore's interior-point solver.

![Relative duality gap on a logarithmic scale against fit time in seconds on a
linear scale for Shrinkage, Linfa, and SmartCore.](figures/elastic-net.png)

*Elastic net convergence with 100 observations, 10,000 features, and 10 planted
signals. Each marker shows the median fit time and relative duality gap across
up to five independent fits at one solver setting. Horizontal bars span the
middle half of measured fit times.*

## Problem and accuracy

The dataset has 100 observations and 10,000 features in correlated groups of
eight. The generator uses seed `20261026` and forms the response from 10
features spaced evenly across the design, with coefficients alternating between
`+1` and `-1`. It adds uniform noise and centers the response to give SmartCore
a zero intercept. All three solvers receive the same column-major `f64` design
and response, start from zero coefficients, and run in one thread with feature
normalization disabled. Shrinkage and SmartCore use `ndarray` 0.17. Linfa uses
`ndarray` 0.16.

All three solvers target the following objective with $\lambda = 0.125$ and
$\alpha = 0.5$.

$$
  \frac{\lVert y - X\beta\rVert_2^2}{2n}
  + \lambda\left( \alpha\lVert\beta\rVert_1 + \frac{1 - \alpha}{2}\lVert\beta\rVert_2^2 \right).
$$

A separate Rust evaluator recomputes the objective, duality gap, and maximum
coordinate KKT violation from the data and returned coefficients. It uses the
residuals divided by the sample count as the dual point and divides the gap by
the null-model loss $\lVert y\rVert_2^2/(2n)$. Metewand ends each repetition
after the first fit with a relative duality gap of at most $10^{-8}$, when it
exhausts the schedule, or when it hits the time limit.

Metewand runs Shrinkage and Linfa with progressively tighter solver tolerances
from `1e-1` through `1e-9`. Shrinkage's solver checks absolute KKT violation.
Linfa checks relative coefficient updates and its own duality gap. [Linfa
stopping
rules](https://docs.rs/crate/linfa-elasticnet/0.8.1/source/src/algorithm.rs)

SmartCore's profile uses caps of 1, 4, and 16 interior-point iterations. Its
adapter converts the penalty strengths to the shared objective's scale and uses
implicit L2 augmentation. SmartCore builds a dense Gram matrix during each fit.
[SmartCore elastic net
implementation](https://github.com/smartcorelib/smartcore/blob/f39624c64742e2450455ad04ae0a2899eb309e69/src/linear/elastic_net.rs)

Shrinkage and Linfa reached the target in all five recorded repetitions.
SmartCore exhausted its schedule with a relative duality gap of
$6.6 \times 10^{-4}$ at the 16-iteration setting.

## Timing and reproduction

Metewand runs the measurements serially and discards one warm-up per worker. It
randomizes the order of Shrinkage and Linfa fits. The worker times model
construction and the public fit call in release mode. Linfa computes fitted
values inside `fit` and skips coefficient variance estimation when the design
has more features than observations. Dataset loading, coefficient extraction,
result serialization, and independent evaluation happen outside the fit timer.

The saved measurements include the hardware, compiler and library versions,
dataset and worker hashes, each fit's timing and accuracy, and why each
repetition stopped. The figure describes one synthetic dataset at one
regularization strength on the recorded machine. [Saved
measurements](figures/elastic-net.json)

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
