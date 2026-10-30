# Linear regression

A Gaussian linear model estimates a real-valued conditional mean. With a
complete penalty $P$, its objective is

$$
  \min_{\beta_0,\beta}\;\frac{\|y - \beta_0\mathbf{1} - \widetilde X\beta\|_2^2}{2n}
  + P(\beta).
$$

Choose [L1](../penalties/l1.md) for sparsity, [ridge](../penalties/ridge.md) for
quadratic shrinkage, or another compatible penalty. The response must contain
one finite real value per observation.

## Fit an elastic net

```rust
use faer::{Col, Mat};
use shrinkage::{ElasticNet, Gaussian, Problem, ProximalGradient, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(6, 2, |i, j| if j == 0 { i as f64 } else { (i % 2) as f64 });
    let y = [1.0, 3.0, 4.0, 7.0, 8.0, 11.0];
    let fit = Problem::new(&x, Gaussian::new(&y), ElasticNet::new(0.1, 0.2))
        .fit_with(&ProximalGradient::<Col<f64>>::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Original-scale coefficients: {:?}", fit.coefficients());
    println!("Response means: {:?}", fit.predict(&x)?);
    Ok(())
}
```

The default fits an unpenalized intercept and standardizes features using
training statistics. See
[normalization](../modeling.md#normalization-and-predictions) to change these
modeling choices.

For Gaussian L1 problems, [coordinate
descent](../reference/solvers/coordinate-descent.md) and the [Lasso
builder](lasso.md) provide a specialized implementation.
[IRLS](../reference/solvers/irls.md) also supports separable penalties; the
[hybrid solver](../reference/solvers/hybrid.md) handles supported fusion and
SLOPE models. The [compatibility table](../reference/solvers/index.md) lists the
available combinations.
