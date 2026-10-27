# Shrinkage for Python

Shrinkage fits dense, single-response Gaussian lasso models through Rust
coordinate descent. Fitted results provide original-scale coefficients and
predictions, training preprocessing, and convergence diagnostics. The Python
package copies NumPy inputs into independent Rust storage and releases the GIL
during fitting and prediction.

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

## Gaussian lasso

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
  \frac{1}{2n}\|y - \widetilde X\theta
  - \widetilde b\mathbf1\|_2^2 + \lambda\|\theta\|_1.
$$

The intercept is unpenalized. Penalties and diagnostics use normalized
coefficients `theta`, while `fit.coefficients` and `fit.intercept` use the
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
The first Python slice supports lasso fits and prediction; paths and other
penalties remain later milestones.

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
evaluated objectives and KKT conditions, input ownership, concurrent Python
execution during fitting, and all diagnostics from 21 Rust fixtures.

Regenerate the fixtures after an intentional numerical change:

```console
cargo run --locked --example python_lasso_fixture --features ndarray_v0_17 > packages/python/tests/fixtures/lasso_rust.json
```

`task example` and CI run that generator with `--check` to verify the committed
fixtures against the current Rust implementation.

See [the documentation](https://shrinkage.cc/) for the Rust API, development
instructions, and independent Rust and Python release conventions.
