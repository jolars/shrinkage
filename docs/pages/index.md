# Shrinkage

## Explore regularization

::: {.component name="optimization-explorer"}

![Gaussian loss contours overlaid with the L1 penalty region, alongside contours
of the penalized objective.](figures/optimization-explorer.svg)

Combine a [datafit](datafits/index.md), a [penalty](penalties/index.md), and a
[compatible solver](reference/solvers/index.md) to explore a two-coefficient
model. Enable JavaScript to adjust the problem and follow the solver toward its
optimum. The penalty region uses the budget at the penalized optimum, showing
the equivalent constrained problem.

:::

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

Browse the Guide's [datafits](datafits/index.md), [models](models/index.md), and
[penalties](penalties/index.md) for mathematical definitions, constraint plots,
and model examples. The [solver reference](reference/solvers/index.md) compares
supported compositions and convergence diagnostics.

The Rust API supports regularization paths and penalties including lasso, ridge,
elastic net, [SLOPE](models/slope.md), disjoint Group LASSO, [clustered
lasso](models/clustered-lasso.md), and [chain fused
lasso](models/chain-fused-lasso.md). SLOPE and both fusion penalties have dense
Gaussian and logistic hybrid solvers. See the [Rust API
reference](https://shrinkage.cc/packages/rust/) for supported combinations and
the [statistical
conventions](https://github.com/jolars/shrinkage/blob/main/DESIGN.md) for loss
normalization, penalty scales, and intercept handling.

The [Python API](https://shrinkage.cc/packages/python/) provides dense Gaussian
lasso, prediction, fitted preprocessing, and convergence diagnostics. Both
packages share numerical implementations and statistical conventions, with
independent versions and release schedules.

To contribute, read the [contribution guide](contribution.md).
