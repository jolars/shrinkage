# Shrinkage <img src="https://raw.githubusercontent.com/jolars/shrinkage/refs/heads/main/assets/logo.png" align="right" width="160" alt="A shrink ray reducing beta coefficients" />

[![CI](https://github.com/jolars/shrinkage/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/shrinkage/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/shrinkage.svg)](https://crates.io/crates/shrinkage)
[![docs.rs](https://img.shields.io/docsrs/shrinkage)](https://docs.rs/shrinkage)

Shrinkage is a Rust library for fitting regression models with penalties that
shrink coefficients, select features, or encourage structure. It supports linear
and logistic regression, dense and sparse matrices, and fitting across a range
of penalty strengths. Start with `Lasso` for linear regression with feature
selection, or combine losses, penalties, and solvers to build other models.

See the [documentation](https://shrinkage.cc/) for installation, the Rust API
reference, and contributor guides.

## Installation

Add [shrinkage](https://crates.io/crates/shrinkage) from crates.io and enable
the feature for your matrix library. For example, to use faer:

```console
cargo add shrinkage --features faer
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

By default, `Lasso` standardizes features and fits an intercept without
penalizing it. The penalty applies to standardized coefficients, while the
returned coefficients and intercept use the original input scale. Prediction
uses the preprocessing learned during fitting. Check `termination()` to see
whether the solver converged or reached its iteration limit.

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

The [examples](examples) cover more models, sparse input, and regularization
paths. Run one from a checkout with:

```console
cargo run --locked --example regularization_path --features faer
```

## Supported models

- **Linear and logistic regression**, plus multivariate linear regression for
  multiple responses.
- **Lasso, ridge, and elastic net** for feature selection and coefficient
  shrinkage.
- **Structured penalties:** [Group LASSO](examples/group_lasso.rs) selects
  groups of features, [SLOPE](https://shrinkage.cc/models/slope/) applies ranked
  penalties, [clustered lasso](https://shrinkage.cc/models/clustered-lasso/)
  encourages equal coefficient magnitudes, and [chain fused
  lasso](https://shrinkage.cc/models/chain-fused-lasso/) encourages neighboring
  coefficients to be equal.
- **Regularization paths** fit a sequence of penalty strengths, reusing previous
  fits to speed up computation.

Solvers include coordinate descent, proximal gradient, and iteratively
reweighted least squares (IRLS). Supported combinations vary by loss and
penalty. The [Rust API reference](https://shrinkage.cc/packages/rust/) covers
these combinations and how to add custom losses and penalties. See the [binomial
example](examples/binomial.rs) for logistic regression.

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

Read the [contribution guide](CONTRIBUTING.md) and the [development
guide](https://shrinkage.cc/development/) for setup and focused checks. Enter
the pinned development environment and run the complete check with:

```console
devenv shell
task check
```

## License

Shrinkage is licensed under the [MIT license](LICENSE).
