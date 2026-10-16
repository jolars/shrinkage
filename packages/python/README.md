# Shrinkage for Python

Shrinkage provides regularized statistical models backed by Rust. The package
provides an installable native extension, version reporting, and a validated
dense NumPy input boundary. Fitting and prediction will follow in the next
milestone. Input preparation is currently private to the native extension.

The distribution and import names are both `shrinkage`. Build from the
repository with Rust 1.89 or newer and standard CPython 3.11 or newer:

```console
uv sync --locked --project packages/python
uv run --locked --project packages/python python -c "import shrinkage; print(shrinkage.__version__)"
```

The compatibility target is CPython 3.11–3.15 on Linux x86_64, Windows x86_64,
and macOS arm64 and x86_64. CI builds and imports an interpreter-specific wheel
for each combination. Free-threaded Python and other interpreters are outside
the initial support matrix. Wheels are not published yet.

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
Do not mutate or resize the arrays, their owners, or aliases concurrently during
preparation, including from already-running native NumPy operations. Once
preparation returns, the arrays may be changed or deleted without affecting the
snapshots. Preparation never changes input values, flags, or ownership.

The native container has no Python references and is immutable from Python. This
establishes ownership for the next fitting milestone to release the GIL. The
buffer API copies strided values without Python indexing, and finite-value
checks run in Rust. Python callbacks stay outside numerical loops. The private
native test interface is not a supported fitting API.

See [the documentation](https://shrinkage.cc/) for the Rust API, development
instructions, and independent Rust and Python release conventions.
