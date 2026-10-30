# Shrinkage for Python

Shrinkage fits dense, single-response Gaussian and binomial models through Rust.
Compose clustered lasso, chain fused lasso, or SLOPE with an explicit hybrid or
proximal-gradient solver. Gaussian lasso also has a convenience API using
coordinate descent. Fitted results provide original-scale coefficients and
predictions, training preprocessing, and convergence diagnostics. The package
copies NumPy inputs into independent Rust storage and releases the GIL during
fitting and prediction.

The distribution and import names are both `shrinkage`. Build from the
repository with Rust 1.85 or newer and standard CPython 3.11 or newer:

```console
uv sync --locked --project packages/python
uv run --locked --project packages/python python -c "import shrinkage; print(shrinkage.__version__)"
```

The compatibility target is CPython 3.11–3.15 on Linux x86_64, Windows x86_64,
and macOS arm64 and x86_64. CI tests Python 3.11 and 3.15 on each platform,
including building and importing an interpreter-specific wheel. Free-threaded
Python and other interpreters are outside the initial support matrix. Wheels are
not published yet.

## Compose a model

```python
import numpy as np
from shrinkage import Binomial, ClusteredLasso, Hybrid, Problem, StoppingCriterion

X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
y = np.array([0.1, 0.2, 0.8, 0.9])
problem = Problem(
    X,
    Binomial(y, fit_intercept=True),
    ClusteredLasso(l1=0.02, fusion=0.03),
    normalization="auto",
)
fit = problem.fit_with(
    Hybrid(stopping=StoppingCriterion.proximal_gradient_mapping(1e-8))
)
assert fit.termination == "converged"
logits = fit.predict(X)
probabilities = np.exp(-np.logaddexp(0, -logits))
```

Use `Gaussian(y)` for average squared loss divided by two, or `Binomial(y)` for
average Bernoulli negative log likelihood. Both fit an unpenalized intercept by
default. Binomial responses must lie in `[0, 1]`; fractional responses denote
proportions with equal observation weights. Binomial `predict` returns logits,
and the conversion above gives probabilities without overflow. Gaussian
predictions are response means. Both have shape `(n_samples,)` and dtype
`float64`.

`Problem(X, datafit, penalty, *, normalization="auto")` retains input references
without copying. `problem.fit_with(solver, *, copy=True)` snapshots the current
inputs on every call. Keep inputs unchanged until fitting returns. Returned
results retain no problem, training arrays, or design snapshots. Array-valued
properties return independent copies, so mutating them cannot change prediction.

Choose one complete penalty, applied to optimization-scale coefficients `beta`.
The intercept is excluded:

  | Penalty                                        | Definition and configuration                                                                           |
  | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
  | `ClusteredLasso(l1, fusion)`                   | `l1 * sum(abs(beta)) + fusion * sum(i < j, abs(beta[i] - beta[j]))`; each signed pair contributes once |
  | `ChainFusedLasso(l1, fusion, *, weights=None)` | `l1 * sum(abs(beta)) + fusion * sum(weights * abs(diff(beta)))`; feature order defines the chain       |
  | `Slope(weights)`                               | Rank weights multiply decreasing coefficient magnitudes; weights include the complete penalty strength |

Strengths and weights must be finite and nonnegative. Chain weights default to
one and otherwise have length `max(n_features - 1, 0)`; zero weights cut the
chain. SLOPE requires one nonincreasing weight per feature. Equal SLOPE weights
give L1, and zero strengths give the corresponding unpenalized limits. Weight
inputs accept one-dimensional real numeric sequences and are copied into
immutable tuples; the stricter dense contract below applies to `X` and `y`.
Strengths, weight lengths, and solver compatibility are checked when fitting.

### Choose a solver

`Hybrid(...)` returns `HybridFit`. Gaussian models alternate PG transitions and
exact block sweeps. Binomial models use weighted hybrid subproblems inside IRLS,
with backtracking on the original logistic objective. Only the full
original-loss mapping, including the free-intercept derivative, certifies
convergence. A surrogate check or unchanged block sweep does not certify it.

All solver arguments are keyword-only. Hybrid defaults match Rust:

  | Control                | Default                     | Meaning                                                                              |
  | ---------------------- | --------------------------- | ------------------------------------------------------------------------------------ |
  | `stopping`             | `None`                      | Absolute full-space PG mapping tolerance `1e-6`                                      |
  | `max_iterations`       | `10_000`                    | Total committed PG transitions plus block sweeps, shared across logistic subproblems |
  | `max_outer_iterations` | `100`                       | Accepted logistic IRLS steps                                                         |
  | `max_inner_iterations` | `1_000`                     | PG transitions plus sweeps per logistic quadratic model                              |
  | `max_outer_backtracks` | `100`                       | Logistic step halvings after the full trial                                          |
  | `cluster_sweeps`       | `5`                         | Block sweeps between PG transitions                                                  |
  | `check_every`          | `10`                        | Check interval, with checks also initially and at the budget                         |
  | `screening`            | `True`                      | Original-problem Gap Safe zero screening                                             |
  | `working_set_policy`   | `WorkingSetPolicy.auto()`   | Independent heuristic restrictions                                                   |
  | `step_policy`          | `HybridStep.backtracking()` | PG curvature policy                                                                  |

Gaussian fits ignore the logistic budgets. Iteration limits, sweep counts, and
check intervals must be positive. Backtracking limits may be zero.
`HybridStep.backtracking(initial_curvature=None, max_backtracks=100)` uses a
positive supplied curvature or a column-based seed and doubles it after rejected
trials. `HybridStep.fixed_bound()` uses a computed Frobenius bound.

Working sets offer `WorkingSetPolicy.auto()`, `.disabled()`, `.ever_active()`,
and `.adaptive(initial_size=10, prune=True)`. Automatic sets resolve to adaptive
sets for single fits. Screening and working sets are independent; disable both
to update all features. A heuristic exclusion is not a certified zero.

Use `ProximalGradient(...)` with the same problem for the reference solver. It
returns `ProximalFit` and optimizes the original Gaussian or binomial loss
directly. Its controls are `stopping=None`, `max_iterations=10_000`,
`initial_step=1.0`, `max_backtracks=100`, and `screening=True`. `max_iterations`
counts accepted proximal updates. The initial step must be finite and positive;
backtracking halves it. Working sets are a Hybrid control.

Both solvers support only `StoppingCriterion.proximal_gradient_mapping(...)`,
with a finite, strictly positive absolute tolerance. Duality-gap and KKT
criteria produce compatibility errors with a supported alternative.

### Inspect the result

Both result types report `solver`, `family`, `coefficients`, `intercept`,
`preprocessing`, `termination`, `iterations`, `objective`, `stopping`,
`stopping_value`, `stopping_threshold`, and `step_size`. Check `termination`:
`"iteration_limit"` returns finite parameters without certifying convergence.
`dual_certificate`, `dual_objective`, and `duality_gap` are optional
original-loss diagnostics and are `None` when screening is disabled. The padded
duality gap does not replace mapping stopping. `screening_diagnostics` reports
certificate passes, zero tests, certified features, remaining dimension,
rebuilds, and fallbacks.

`HybridFit` also reports `pg_steps`, `coordinate_sweeps`, `diagnostic_checks`,
`backtracks`, `outer_iterations`, `outer_backtracks`, `step_policy`, and
`working_set_diagnostics`. Inner logistic checks concern quadratic models;
original-loss checks control convergence. Outer counters are zero for Gaussian
fits. Working-set diagnostics retain the resolved policy and feature and block
work.

Hybrid cluster metadata uses the optimization scale. `cluster_labels` is an
integer array of shape `(n_features,)` indexing `cluster_values`, whose shape is
`(n_clusters,)`. Clustered lasso values are signed and strictly increasing.
Chain values follow contiguous segment order; equal values across zero-weight
cuts can have separate labels. SLOPE values are decreasing magnitudes, and
`coefficient_signs` has shape `(n_features,)`. Reconstruct SLOPE coefficients as
`coefficient_signs * cluster_values[cluster_labels]`; other penalties use the
indexed values directly. Dividing by the fitted training scales gives the
reported original-scale coefficients. Unequal scales can change equalities.

Run all three penalties with both families and compare both solvers through the
installed package:

```console
uv run --locked --project packages/python python packages/python/examples/hybrid.py
```

## Gaussian lasso convenience API

```python
import numpy as np
from shrinkage import Lasso, StoppingCriterion

X = np.array([[0.0], [1.0], [2.0]])
y = np.array([1.0, 3.0, 5.0])
fit = Lasso(
    0.1,
    normalization="auto",
    stopping=StoppingCriterion.duality_gap(1e-8),
).fit(X, y)
assert fit.termination == "converged"
predictions = fit.predict(X)
np.testing.assert_allclose(predictions, X @ fit.coefficients + fit.intercept)
```

`Lasso(lambda_, *, fit_intercept=True, normalization="auto", stopping=None, max_iterations=10_000)`
configures a fit. `fit(X, y, *, copy=True)` returns an owned `LassoFit`; it
leaves the configuration and inputs unchanged. The penalty strength must be
finite and nonnegative. The objective is

$$
  P(\beta_0,\beta) = F(\beta_0,\beta) + R(\beta)
                   = \frac{1}{2n}\|y - \widetilde X\beta
                     - \beta_0\mathbf1\|_2^2 + \lambda\|\beta\|_1.
$$

Here $F$ is the datafit and $R$ is the complete penalty. For a feasible dual
variable $\theta$, $D(\theta)$ denotes the dual objective, and
$G(\beta_0,\beta,\theta) = P(\beta_0,\beta) - D(\theta)$ is the duality gap.

The intercept is unpenalized. Penalties and diagnostics use normalized
coefficients `beta`, while `fit.coefficients` and `fit.intercept` use the
original input scale. `fit.predict(X_new)` returns a `float64` NumPy array of
shape `(n_samples,)` using raw columns in training order. Empty prediction
batches are allowed. Prediction uses the fitted parameters without estimating
new centers or scales.

Normalization presets match Rust:

  | Preset          | Centering                                      | Scaling                       |
  | --------------- | ---------------------------------------------- | ----------------------------- |
  | `"auto"`        | Mean when fitting an intercept; otherwise none | Population standard deviation |
  | `"none"`        | None                                           | None                          |
  | `"center"`      | Mean                                           | None                          |
  | `"standardize"` | Mean                                           | Population standard deviation |
  | `"min_max"`     | Minimum                                        | Range                         |
  | `"max_abs"`     | None                                           | Maximum absolute value        |
  | `"l1"`          | None                                           | L1 norm                       |
  | `"l2"`          | None                                           | L2 norm                       |

Explicit presets apply independently of `fit_intercept`. Explicit centering
without fitting an intercept can induce a fixed original-scale intercept, which
the result retains. Zero computed scales become one. `fit.preprocessing` reports
resolved `centering` and `scaling` rules and the fitted `centers` and `scales`.
Each enabled vector has shape `(n_features,)`; disabled vectors are `None`.
Coefficients have shape `(n_features,)`. Returned arrays are independent copies,
so changing them does not change prediction or the native fit.

## Convergence and errors

Always inspect `fit.termination`. `"converged"` certifies the selected stopping
check on refreshed state. `"iteration_limit"` returns finite parameters and
diagnostics after the sweep budget expires; it does not certify convergence or
raise an exception. `fit.iterations` counts complete coordinate sweeps; an
initially optimal fit takes zero sweeps. The budget must be positive.

With `stopping=None`, positive penalties use relative duality gap `1e-6`, and
zero penalty uses absolute KKT violation `1e-6`. Set either check explicitly:

```python
gap = StoppingCriterion.duality_gap(1e-8, absolute=1e-12)
kkt = StoppingCriterion.kkt_violation(1e-10)
```

The gap threshold is `absolute + relative * fit.reference_loss`, using the fixed
average squared loss at zero coefficients and the optimal enabled intercept. A
zero reference loss gives a threshold equal to `absolute`. KKT violation is the
largest absolute normalized-design coefficient violation, including the
intercept derivative when fitted. All tolerances must be finite and nonnegative,
with at least one positive component. KKT uses only an absolute tolerance.

Results expose `stopping`, `stopping_value`, `stopping_threshold`, `objective`,
`kkt_violation`, `dual_certificate`, `dual_objective`, `duality_gap`, and
`reference_loss`. The dual certificate has shape `(n_samples,)`. At zero penalty
the residual-based dual certificate can give a loose gap, which is why KKT is
the default there.

Invalid array types and dtypes raise `TypeError`. Invalid dimensions, values,
penalty strengths, normalization names, stopping controls, and copying policies
raise `ValueError`. Finite inputs that produce nonfinite arithmetic raise
`FloatingPointError`, retaining the Rust operation and iteration in the message.
Line-search exhaustion raises `RuntimeError` with the iteration and solver
settings. Python paths, cross-validation, and plotting remain later milestones.

## Dense input contract

The package requires NumPy 2.0 or newer. A design matrix `X` must be a
two-dimensional `numpy.ndarray` with shape `(n_samples, n_features)`. Its
single-response target `y` must be a one-dimensional `numpy.ndarray` with shape
`(n_samples,)`. Both arrays must have native-endian `float64` dtype, aligned
storage, and finite values. At least one observation is required; zero features
are allowed. Column-vector targets are rejected rather than squeezed.

The boundary accepts exact `numpy.ndarray` instances. Lists, sparse matrices,
memoryviews, array protocol objects, and ndarray subclasses (including masked
arrays and memory-mapped subclasses) raise `TypeError`. Convert these explicitly
when appropriate, and resolve masks before conversion:

```python
import numpy as np

X = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
y = np.asarray([1, 2], dtype=np.float64)
```

Other dtypes, including `float32`, integers, objects, complex values, and
non-native byte order, raise `TypeError`. The boundary never casts them
implicitly. Invalid dimensions, lengths, empty observations, nonfinite values,
or unaligned storage raise `ValueError`. Use
`np.require(array, dtype=np.float64, requirements=["A", "E"])` to obtain an
aligned base array explicitly.

  | NumPy storage                              | Input preparation                                          |
  | ------------------------------------------ | ---------------------------------------------------------- |
  | C-contiguous or Fortran-contiguous         | Copy `X` into Rust column order and `y` into a Rust vector |
  | Transposed, sliced, reversed, or broadcast | Copy logical values using the array's strides              |
  | Writable or read-only, owned or a view     | Use the same independent snapshot policy                   |
  | Any input with `copy=False`                | Raise `ValueError`; NumPy zero-copy input is unsupported   |

Each accepted array is copied once, including read-only arrays. A read-only flag
does not prevent a writable alias from changing the same storage. There are no
accepted zero-copy NumPy inputs. The snapshots require
`8 * (n_samples * n_features + n_samples)` bytes in addition to the original
arrays, excluding allocation metadata.

Preparation retains the array owners and holds the GIL until copying finishes.
The caller must prevent concurrent mutation or resizing through aliases or
already-running native operations during this phase. Because `fit` and `predict`
do not signal the end of copying, keep their inputs unchanged until the call
returns. Other Python threads can run during the subsequent numerical work.
Changing or deleting inputs after return cannot affect the fitted model.
Preparation never changes input values, flags, or ownership.

Prediction follows the same contract, except it accepts zero rows and needs no
response vector. Its snapshot requires `8 * n_samples * n_features` bytes. Input
buffers are released before the GIL is released. The native snapshots retain no
Python references, and the result retains no training matrix or response. The
buffer API copies strided values without Python indexing, and finite-value
checks run in Rust. Python callbacks stay outside numerical loops.

## Validation

`task python:check` tests the editable package. `task python:artifacts` runs the
same tests against an installed wheel and a wheel rebuilt outside the checkout
from a source distribution. These include analytical solutions, independently
evaluated objectives and analytical Gaussian and logistic solutions, input
ownership, concurrent fits, GIL release during fitting and prediction, and all
diagnostics from public Rust fixtures. The model fixtures cover all twelve
combinations of penalty, family, and solver, every normalization preset, both
Hybrid step policies, working sets, screening settings, and iteration limits.

Regenerate the fixtures after an intentional numerical change:

```console
cargo run --locked --example python_lasso_fixture --features ndarray_v0_17 > packages/python/tests/fixtures/lasso_rust.json
cargo run --locked --example python_model_fixture --features ndarray_v0_17 > packages/python/tests/fixtures/model_rust.json
```

`task example` and CI run that generator with `--check` to verify the committed
fixtures against the current Rust implementation.

See [the documentation](https://shrinkage.cc/) for the Rust API, development
instructions, and independent Rust and Python release conventions.
