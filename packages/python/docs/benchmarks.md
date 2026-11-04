# Lasso benchmark

This comparison shows how quickly the Python APIs of shrinkage, scikit-learn,
and skglm reduce the lasso duality gap on one dense problem at one
regularization strength.

![Lasso convergence for shrinkage, scikit-learn, and skglm, with linear fit time
on the horizontal axis and duality gap on the logarithmic vertical
axis.](figures/lasso.png)

The horizontal axis shows fit time on a linear scale. The vertical axis shows
the independently computed duality gap on a logarithmic scale, so convergence
over several orders of magnitude remains visible. Each marker shows the median
time and gap of five measurements at the same iteration limit. Horizontal error
bars span the middle half of the times.

Metewand increases each API's iteration limit from 1 through 40. Shrinkage
sweeps its current working set, and scikit-learn sweeps all features. skglm
counts working-set subproblems, each allowing up to 50,000 inner coordinate
sweeps with Anderson acceleration. Equal limits therefore represent different
amounts of work. Every fit starts from zero coefficients. Each curve ends when
its independently checked gap meets the accuracy target. Each point's time
measures that fit from initialization through its selected limit.

The figure uses Metewand's in-worker timer around model construction and the
call to `fit`. This includes input validation, the copies required by each API,
and numerical fitting. Loading the dataset, importing packages, extracting
coefficients, serializing results, and checking accuracy happen outside that
timer.

## Problem and accuracy

The dataset has 5,000 observations and 1,000 independent standard Gaussian
features. The response uses 20 nonzero coefficients, alternating between 1 and
−1, plus standard Gaussian noise. The manifest fixes NumPy's dataset seed at
`7363749798296254`, preserving the original dataset derived from experiment seed 42.
This keeps the arrays stable when the benchmark runner or software changes. We
center the features and response once before fitting and pass all three methods
the same Fortran-contiguous NumPy `float64` arrays.

All three methods minimize

$$
  \frac{1}{2n}\lVert y - X\beta\rVert_2^2
  + \alpha\lVert\beta\rVert_1, \qquad \alpha
  = 0.1\,\frac{\lVert X^T y\rVert_\infty}{n}.
$$

The plotted gap is the primal objective minus a feasible dual objective. It
provides an upper bound on the objective's suboptimality at the returned
coefficients.

We disable intercept fitting and shrinkage's default column standardization.
scikit-learn uses cyclic coordinate descent without a precomputed Gram matrix.
Shrinkage and scikit-learn copy their inputs; skglm can reuse the supplied
arrays. skglm uses its `Lasso` API with the default initial working-set size of
10 and subdifferential-based working-set selection.

The common target is a duality gap no larger than `1e-8` times the null-model
loss `||y||² / (2n)`. shrinkage uses
`StoppingCriterion.duality_gap(relative=1e-8)`. scikit-learn uses `tol=5e-9`
because its gap threshold scales with `||y||² / n`; it also checks coordinate
updates. skglm uses `tol=1e-10` for its maximum subdifferential violation, which
differs from a duality-gap threshold. The common evaluator determines when its
curve ends. See the [skglm Lasso
documentation](https://contrib.scikit-learn.org/skglm/generated/skglm.Lasso.html)
for its objective and iteration controls.

An independent NumPy evaluator recomputes the objective, a feasible dual bound,
and the KKT violation from the returned coefficients. Every plotted curve must
end at the common gap target. Earlier points retain their larger gaps. See
scikit-learn's [Lasso
documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
for its objective and stopping rules.

## Measurement and reproduction

Shrinkage checks full convergence at initialization, every ten coordinate
sweeps, and at the iteration limit. Working sets update at these checks. The
initial budget uses the number of initial KKT violations, bounded between 10 and
64 features. The profile allows up to 40 sweeps to accommodate this cadence.

On Linux, shrinkage copies matrices of at least 4 MiB into anonymous Rust-owned
mappings with transparent huge-page advice. Smaller inputs, other platforms, and
unsuccessful mapping or advice requests use ordinary Rust allocations. Both
paths preserve the same input-copy and ownership contract. NumPy also [uses
huge-page advice on
Linux](https://numpy.org/doc/stable/reference/global_state.html#madvise-hugepage-on-linux).

The run uses a release build of shrinkage with one code-generation unit, one
thread per numerical library, serial execution, one discarded warm-up per
worker, and five measured profiles per implementation. Metewand's
`fresh_sequence` profile supports these fits at increasing iteration limits
without adding callbacks to the solvers. It randomizes method order within
comparison blocks. Workers retain their imports, but every fit creates a new
model. skglm's Numba compilation occurs during the discarded warm-up, so the
figure describes fits with compiled kernels. A single `compute bench`
reservation covers all three methods.

The [retained measurements](figures/lasso.json) are Metewand's JSON analysis
export. They contain complete observation metrics, effective settings, profile
progress, and recorded provenance. Shared provenance identifies the CPU,
operating system, source revision and dirty status, Metewand executable hash,
dataset seed, and dataset content hash. Worker statistics identify Python and
numerical package versions and the native-extension hash. The export also
retains Metewand's primary `execute_only` wall time, which includes result
conversion and protocol work.

Metewand checks declared repetitions and profile completion before exporting.
The figure uses selected successful measurements, including valid intermediate
points with larger gaps. A successful retry can supply those measurements after
an earlier failure. The full run directory retains the attempt history.

From the repository root, with the Metewand CLI installed:

```sh
compute bench --reason "Python lasso versus scikit-learn and skglm" -- devenv shell -- task python:bench
```

The [benchmark sources and setup
instructions](https://github.com/jolars/shrinkage/tree/main/packages/python/benchmarks)
explain CLI installation and figure regeneration. Documentation builds use the
saved figure and do not execute the benchmark.

These timings describe one simulated dataset on one machine. Other matrix sizes,
feature correlations, regularization strengths, and real datasets may change the
comparison. This benchmark does not measure regularization paths, sparse inputs,
cross-validation, or memory use.
