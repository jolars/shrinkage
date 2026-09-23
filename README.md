# Shrinkage

Shrinkage is a Rust library for regularized statistical models. It currently
fits Gaussian lasso models with dense or sparse CSC input through a statically
dispatched coordinate-descent solver. Compositional and runtime APIs are planned
for later milestones. See [DESIGN.md](DESIGN.md) for the architecture and
statistical conventions, and [TODO.md](TODO.md) for the implementation
checklist.

## Gaussian lasso

Enable the `faer` feature for this example:

```rust
use faer::Mat;
use shrinkage::{Lasso, Termination};

let x = Mat::from_fn(3, 1, |i, _| i as f64);
let fit = Lasso::new(0.1).fit(&x, &[1.0, 3.0, 5.0])?;
assert_eq!(fit.termination(), Termination::Converged);
let predictions = fit.predict(&x)?;
```

`Lasso::new(lambda)` defaults to an unpenalized intercept, training-column means
and population standard deviations, an absolute KKT tolerance of `1e-6`, and a
limit of 10,000 coordinate sweeps. The objective is

$$
  \frac{1}{2n}\|y - \widetilde X\theta
  - \widetilde b\mathbf1\|_2^2 + \lambda\|\theta\|_1.
$$

The penalty acts on the normalized coefficients. `coefficients()`,
`intercept()`, and `predict()` use the original input scale. Fitted centers and
scales remain available through `preprocessing()`; prediction does not estimate
new statistics.

Choose normalization with `.normalize(Normalization::...)`:

  | Option                     | Centering                                      | Scaling                       |
  | -------------------------- | ---------------------------------------------- | ----------------------------- |
  | `Auto` (default)           | Mean when fitting an intercept; otherwise none | Population standard deviation |
  | `None`                     | None                                           | None                          |
  | `Center`                   | Mean                                           | None                          |
  | `Standardize`              | Mean                                           | Population standard deviation |
  | `MinMax`                   | Minimum                                        | Range                         |
  | `MaxAbs`                   | None                                           | Maximum absolute value        |
  | `L1`                       | None                                           | L1 norm                       |
  | `L2`                       | None                                           | L2 norm                       |
  | `Custom { center, scale }` | Selected independently                         | Selected independently        |

Custom choices reuse LazyMatrix's enums, re-exported from Shrinkage:

```rust
use shrinkage::{Centering, Lasso, Normalization, Scaling};

let model = Lasso::new(0.1).normalize(Normalization::Custom {
    center: Centering::Mean,
    scale: Scaling::L2,
});
```

Scales based on norms or maximum absolute values are computed after centering.
Use `Normalization::None` for the raw design. Only `Auto` changes centering when
`.fit_intercept(false)` is selected. Explicit centering is honored even without
a fitted intercept, and its induced original-scale intercept is retained.
Builder call order does not change this behavior. Fitted preprocessing exposes
the resolved `centering()` and `scaling()` rules as well as their values.
Computed zero scales become one, and zero-norm normalized columns stay at zero.
Inputs must be finite, and the solver does not impute missing values.

A successful `Result` contains a finite fit, which may have reached the
iteration limit. Check `termination()` before treating it as converged. The
result reports `iterations()`, `objective()`, and `kkt_violation()`. The KKT
check uses the averaged loss gradient and includes the intercept condition; it
is confirmed on reconstructed residuals before reporting convergence. Invalid
input, numerical failures, and backend preprocessing failures return distinct
`LassoError` variants, with backend error sources preserved.

The solver keeps sparse centering in a scalar residual offset. Coordinate dots
and updates touch stored column entries, with residual refreshes every 50 sweeps
and before final reporting. Solver storage is `O(n + p)`; fitting does not
materialize a normalized design or form a Gram matrix.

Run the dense and CSC fitting example:

```console
cargo run --locked --example lasso --features faer
```

## Matrix dependency

The crate pins LazyMatrix 0.3.0 from crates.io. No matrix backend is enabled by
default. The optional `faer` feature selects faer 0.24, and `nalgebra` selects
nalgebra 0.34 with nalgebra-sparse 0.11 through LazyMatrix's version-specific
features. Custom matrices can implement LazyMatrix's `RawColumns<f64>` and
`ColumnStats<f64>` capabilities.

Run the dense/sparse normalization example:

```console
cargo run --locked --example normalization --features faer
```

The example checks that lazy products preserve predictions after transforming
coefficients and the intercept back to the original scale. It does not fit a
model.

## Development

Enter the development environment and run the complete local check:

```console
devenv shell
task check
```

The toolchain is pinned to Rust 1.89.0, which is also the minimum supported Rust
version. `Cargo.lock` and `devenv.lock` are tracked for reproducible
development. CI runs the same Rust checks, example, and Panache checks as
`task check`.

Useful focused commands are:

```console
task format
task lint
task test
task bench
```

`task bench` measures end-to-end dense and CSC fitting time, including
validation and preprocessing, with only the faer backend enabled. Dataset
construction is outside the timed region. Allocation and memory profiling remain
later work.

Modules use `name.rs` and `name/child.rs`; the repository does not use `mod.rs`.

## Releases

Versionary prepares release pull requests after CI passes on `main` and manages
`CHANGELOG.md`. Before enabling releases on GitHub, configure the
`RELEASE_TOKEN` repository secret with access to contents, pull requests, and
issues. Registry publication is not configured.

## License

MIT. See [LICENSE](LICENSE).
