# Multivariate Gaussian

`MultivariateGaussian` fits $q$ real-valued responses with shared observations:

$$
  L(Y,H) = \frac{\|Y - H\|_F^2}{2n}
         = \frac1{2n}\sum_{r=1}^q\sum_{i=1}^n(Y_{ir} - H_{ir})^2.
$$

The loss sums over responses and averages over observations. It never divides by
$q$. Independent response fits therefore use the same penalty strengths as the
corresponding single-response Gaussian problems.

## Evaluate the loss

Store response columns contiguously. Entry $(i,r)$ has index `r * nobs + i`. The
supplied dimensions must be positive and match the response length.

```rust
use shrinkage::MultivariateGaussian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let y = [1.0, 3.0, 2.0, 4.0];
    let datafit = MultivariateGaussian::new(&y, 2, 2);
    let mut gradient = [0.0; 4];
    datafit.gradient(&[0.0; 4], &mut gradient)?;
    assert_eq!(gradient, [-0.5, -1.5, -1.0, -2.0]);
    Ok(())
}
```

Every response has a free, unpenalized intercept by default.
`.fit_intercept(false)` disables all free intercepts. Predictors and predictions
use the same response-major layout as the observations.

Fitting uses [proximal gradient](../solvers/proximal-gradient.md) and an
[AxisPenalty](../penalties/axis.md) with an explicit feature-by-response
`CoefficientLayout`. See [multivariate regression](../models/multivariate.md)
for a complete example and feature selection across responses.

API:
[`shrinkage::MultivariateGaussian`](https://shrinkage.cc/packages/rust/reference/7fdf5e23960fc73db0e87f24ddb2a99c31ff7cb273565e92aaae35a4fcc0bf0a).
