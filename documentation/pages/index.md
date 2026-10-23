# Shrinkage

## Start with lasso

Fit a Gaussian lasso model with the Rust convenience API:

```rust
use faer::Mat;
use shrinkage::Lasso;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let y = [1.0, 3.0, 5.0];

    let fit = Lasso::new(0.1).fit(&x, &y)?;
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
```

By default, `Lasso` fits an unpenalized intercept and standardizes features
using training data. Returned coefficients use the original input scale, and
prediction reuses the fitted preprocessing. Check `fit.termination()` to inspect
convergence.

Follow the [installation guide](installation.md) to enable the `faer` feature
and run the example. Shrinkage is under development; the Rust and Python
packages are available from source and have not yet been published to package
registries.

## Explore further

The Rust API supports regularization paths and penalties including lasso, ridge,
elastic net, SLOPE, and disjoint Group LASSO. See the [Rust API
reference](https://shrinkage.cc/packages/rust/) for supported combinations and
the [statistical
conventions](https://github.com/jolars/shrinkage/blob/main/DESIGN.md) for loss
normalization, penalty scales, and intercept handling.

The [Python API](https://shrinkage.cc/packages/python/) provides dense Gaussian
lasso, prediction, fitted preprocessing, and convergence diagnostics. Both
packages share numerical implementations and statistical conventions, with
independent versions and release schedules.

To contribute, read the [development guide](development.md) and [release
conventions](releases.md).
