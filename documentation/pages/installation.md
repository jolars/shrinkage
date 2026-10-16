# Installation and compatibility

## Rust

Shrinkage requires Rust 1.89 or newer. Until the first crates.io release, use a
checkout of the repository. The default build enables no matrix backends and has
no Python dependencies. Select the versioned feature matching your matrix
library:

  | Feature          | Integration                            |
  | ---------------- | -------------------------------------- |
  | `faer_v0_24`     | faer 0.24                              |
  | `nalgebra_v0_34` | nalgebra 0.34 and nalgebra-sparse 0.11 |
  | `ndarray_v0_17`  | ndarray 0.17                           |
  | `sprs_v0_11`     | sprs 0.11 CSC through `SprsCsc`        |

The short names `faer`, `nalgebra`, `ndarray`, and `sprs` are aliases. Optional
`experimental-block-reader` and `zarrs_v0_22` features enable the provisional
buffered file-backed consumer. The documentation reference includes these
optional capabilities.

From a checkout, run the existing fitting example:

```console
devenv shell -- cargo run --locked --example lasso --features faer_v0_24
```

## Python

The distribution and import names are both `shrinkage`. The package currently
provides installation and version reporting. Numerical bindings and NumPy
dependencies will be added in later milestones.

The CI matrix builds an interpreter-specific wheel and tests its installation
and import for each supported combination:

  | Interpreter                | Linux  | macOS            | Windows |
  | -------------------------- | ------ | ---------------- | ------- |
  | Standard CPython 3.11–3.15 | x86_64 | arm64 and x86_64 | x86_64  |

Other interpreters, free-threaded CPython, Linux musl, and additional
architectures are outside this initial matrix. These are source-build and CI
targets; portable registry wheels are a later distribution milestone.

Build and install the checkout with Rust 1.89 or newer and a supported Python:

```console
devenv shell
task python:install
uv run --locked --project packages/python python -c "import shrinkage; print(shrinkage.__version__)"
```

The build uses the Rust sources from this repository. Installing an extracted
source distribution also works without the repository because Maturin includes
the required local Rust dependency. Neither package is published to a registry
yet.
