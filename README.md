# Shrinkage <img src="https://raw.githubusercontent.com/jolars/shrinkage/refs/heads/main/assets/logo.png" align="right" width="160" alt="A shrink ray reducing beta coefficients" />

[Documentation](https://shrinkage.cc/) covers the Rust API and the developing
Python package. The optional Python package lives in
[`packages/python`](packages/python/README.md); Rust and Python have independent
versions and release schedules.

The Python package fits dense Gaussian lasso and predicts with original-scale
parameters, fitted preprocessing, and convergence diagnostics. See its [fitting
example and input contract](packages/python/README.md#gaussian-lasso).

Shrinkage is a Rust library for regularized statistical models. It currently
fits Gaussian lasso models with dense or sparse CSC input through coordinate
descent and fits lasso, ridge, elastic net, and SLOPE through proximal gradient.
The typed and runtime APIs also accept external smooth datafits and complete
proximal penalties. See [DESIGN.md](DESIGN.md) for the architecture and
statistical conventions, and [TODO.md](TODO.md) for the implementation
checklist.

See the [documentation](https://shrinkage.cc/) for installation, the Rust API
reference, and contributor guides.

## Installation

Shrinkage is under development and has not yet been published to crates.io. Add
it from GitHub and enable the feature for your matrix library:

```console
cargo add --git https://github.com/jolars/shrinkage --features faer
cargo add faer@0.24
```

The core library requires Rust 1.85 or newer. Some optional matrix backends
require a newer compiler; see the [compatibility
guide](https://shrinkage.cc/installation/).

Python bindings are under development. See the [Python package
guide](packages/python/README.md) for their current status and source
installation.

## Usage

Fit a Gaussian lasso model and predict from a matrix whose rows are observations
and columns are features:

```rust
use faer::Mat;
use shrinkage::{Lasso, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let y = [1.0, 3.0, 5.0];

    let fit = Lasso::new(0.1).fit(&x, &y)?;
    assert_eq!(fit.termination(), Termination::Converged);

    println!("Coefficients: {:?}", fit.coefficients());
    println!("Intercept: {}", fit.intercept());
    println!("Predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
```

By default, `Lasso` fits an unpenalized intercept and standardizes features
using training data. The penalty acts on standardized coefficients; returned
coefficients and intercepts use the original input scale. Prediction reuses the
fitted preprocessing. Check `termination()` to distinguish convergence from an
iteration limit.

Compose a loss, a penalty, and a compatible solver for other models. For
Gaussian elastic net:

```rust
use faer::{Col, Mat};
use shrinkage::{ElasticNet, Gaussian, Problem, ProximalGradient};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let y = [1.0, 3.0, 5.0];

    let fit = Problem::new(&x, Gaussian::new(&y), ElasticNet::new(0.1, 0.2))
        .fit_with(&ProximalGradient::<Col<f64>>::new())?;

    println!("Coefficients: {:?}", fit.coefficients());
    Ok(())
}
```

The [examples](examples) cover dense and sparse fitting, normalization,
multivariate models, and regularization paths. Run one from a checkout with:

```console
cargo run --locked --example regularization_path --features faer
```

## Models and solvers

- **Losses:** Gaussian, binomial (logistic), and multivariate Gaussian.
- **Penalties:** lasso, ridge, elastic net, and SLOPE, with scaling, masking of
  unpenalized coefficients, and explicit penalty axes for multivariate models.
- **Solvers:** coordinate descent for Gaussian lasso and proximal gradient for
  smooth convex losses with a complete proximal penalty.
- **Regularization paths:** explicit or automatic penalty sequences, warm
  starts, and screening for Gaussian lasso.

The typed and runtime APIs also accept custom datafits and penalties. See the
[Rust API reference](https://shrinkage.cc/packages/rust/) for component
interfaces and supported combinations.

## Matrix backends

No matrix backend is enabled by default. Match your matrix dependency to one of
these features:

  | Feature    | Versioned feature | Supported input                                     |
  | ---------- | ----------------- | --------------------------------------------------- |
  | `faer`     | `faer_v0_24`      | Dense matrices, views, and sparse CSC matrices      |
  | `nalgebra` | `nalgebra_v0_34`  | Dense matrices, views, and nalgebra-sparse 0.11 CSC |
  | `ndarray`  | `ndarray_v0_17`   | Two-dimensional arrays and views                    |
  | `sprs`     | `sprs_v0_11`      | Sparse CSC matrices and views through `SprsCsc`     |

The short names are aliases for the listed versioned features. Matrix access and
normalization use [LazyMatrix](https://crates.io/crates/lazymatrix). The
[installation guide](https://shrinkage.cc/installation/) also covers
experimental file-backed input through Zarr.

## Documentation

- [Documentation website](https://shrinkage.cc/)—installation and API
  references.
- [Examples](examples)—runnable Rust models and matrix integrations.
- [Design](DESIGN.md)—architecture, statistical conventions, and extension
  contracts.
- [Python package](packages/python/README.md)—binding status and installation.

## Contributing

See the [development guide](https://shrinkage.cc/development/) for setup and
focused checks. Enter the pinned development environment and run the complete
check with:

```console
devenv shell
task check
```

## License

Shrinkage is licensed under the [MIT license](LICENSE).
