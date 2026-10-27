# Gaussian

`Gaussian` fits a single real-valued response using averaged squared error:

$$
  L(y,\eta) = \frac{1}{2n}\sum_{i=1}^n(y_i - \eta_i)^2, \qquad \nabla_\eta L
            = \frac{\eta - y}{n},\qquad \nabla_\eta^2 L = \frac{1}{n}I.
$$

Supply one finite response per observation. The constructor borrows the response
slice. By default, the solver fits a free, unpenalized intercept; use
`.fit_intercept(false)` to disable it. Predictions are fitted response means.

## Evaluate the loss

```rust
use shrinkage::{Datafit, Gaussian};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let y = [1.0, 3.0];
    let datafit = Gaussian::new(&y);
    assert!((datafit.value(&[0.0, 2.0])? - 0.5).abs() < 1e-14);
    Ok(())
}
```

For model fitting, see [linear regression](../models/linear.md) or the [Lasso
convenience API](../models/lasso.md). All four built-in solvers support Gaussian
problems with their respective compatible penalties. Because the loss averages
over observations, a penalty strength is not multiplied by the sample size. See
the [shared conventions](../index.md).

API:
[`shrinkage::Gaussian`](https://shrinkage.cc/packages/rust/reference/8f2aa89a892fb708e0e09c276910730b91ed3cb252a10cacd0a458cfbe59be13).
