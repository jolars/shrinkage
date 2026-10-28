# Lasso convenience API

`Lasso` fits the Gaussian L1 model:

$$
  \min_{b,\theta}\;\frac{\|y - b\mathbf{1} - \widetilde X\theta\|_2^2}{2n}
  + \lambda\|\theta\|_1.
$$

It combines `Gaussian`, `L1`, and specialized cyclic coordinate descent. The
builder fits an unpenalized intercept and standardizes features by default.
Returned coefficients and predictions use the original input scale.

## Fit and inspect

```rust
use faer::Mat;
use shrinkage::{Lasso, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let fit = Lasso::new(0.1).fit(&x, &[1.0, 3.0, 5.0])?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Coefficients: {:?}", fit.coefficients());
    println!("Intercept: {}", fit.intercept());
    println!("Predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
```

Set `.normalize(...)`, `.fit_intercept(...)`, `.max_iterations(...)`, and
`.terminate_on(...)` on the builder. `.tolerance(...)` selects an absolute KKT
tolerance. The default instead uses a relative duality-gap tolerance of `1e-6`
when the strength is positive and absolute KKT tolerance `1e-6` at zero.

## Regularization paths

`fit_path(&x, &y, &lambdas)` fits supplied strengths in caller order, reusing
preprocessing and solver storage with warm starts.
`fit_regularization_path(&x, &y, &PathOptions::new())` constructs a decreasing
sequence from the zero-coefficient threshold; `lambda_max(&x, &y)` returns that
threshold. Gap Safe screening and working sets accelerate paths and can be
controlled separately.

Dense and sparse CSC designs use the same statistical conventions. For backend
requirements, see [installation](../installation.md). Use typed [linear
regression composition](linear.md) to select other penalties or solvers.

API:
[`shrinkage::Lasso`](https://shrinkage.cc/packages/rust/reference/15ac14ea52500c664ef11283ff8f8268bb931648138cfb39e6f72b84444fee90)
and
[`shrinkage::LassoFit`](https://shrinkage.cc/packages/rust/reference/d7f3cc1c178006da270ca45d4069f972c99a6c736656b6241882f09fbaf163bc).
