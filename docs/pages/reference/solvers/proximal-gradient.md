# Proximal gradient

`ProximalGradient` combines a smooth datafit with an exact proximal map for the
complete penalty. A step updates the coefficients by

$$
  \theta^+ = \operatorname{prox}_{sP}(\theta - s\nabla_\theta L).
$$

The free intercept takes an ordinary gradient step. Backtracking halves the step
until the smooth loss satisfies its quadratic upper bound. This is the broadest
built-in solver: it supports Gaussian and binomial losses with all seven base
penalties, compatible wrappers, and explicit multivariate layouts.

## Fit Gaussian ridge

```rust
use faer::{Col, Mat};
use shrinkage::{Gaussian, Problem, ProximalGradient, Ridge, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let fit = Problem::new(&x, Gaussian::new(&[1.0, 3.0, 5.0]), Ridge::new(0.1))
        .fit_with(&ProximalGradient::<Col<f64>>::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Coefficients: {:?}", fit.coefficients());
    Ok(())
}
```

## Controls and convergence

Defaults are 10,000 accepted updates, initial step one, at most 100 backtracking
halvings per iteration, and absolute full PG mapping tolerance `1e-6`. Use
`.max_iterations(...)`, `.initial_step(...)`, `.max_backtracks(...)`, and
`.tolerance(...)` to control them. Only PG mapping stopping is supported. The
check includes the intercept derivative when fitted.

The vector type matches the matrix integration: `faer::Col<f64>`,
`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, or `Vec<f64>` for sprs.
Designs need forward and transposed products plus preprocessing capabilities.
The iteration reuses its product and proximal workspaces.

Typed and runtime problems share the solver. Explicit paths accept compatible
penalty sequences; automatic paths additionally need `PathPenalty`.
[Multivariate regression](../../models/multivariate.md) uses the same iteration
with `AxisPenalty` and response-major storage.

API:
[`shrinkage::ProximalGradient`](https://shrinkage.cc/packages/rust/reference/dbf207e0ed64fdc5c0f5cc6c364e567527bca92737c69f45f608ed62e8d669d5).
