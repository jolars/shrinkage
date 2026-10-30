# Proximal gradient

`ProximalGradient` combines a smooth datafit with an exact proximal map for the
complete penalty. A step updates the coefficients by

$$
  \beta^+ = \operatorname{prox}_{sR}(\beta - s\nabla_\beta F(\beta_0,\beta)).
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

## Dynamic zero screening

Gap Safe zero screening is enabled by default for Gaussian and binomial fits
with clustered lasso, chain fused lasso, and SLOPE. Set `.screening(false)` to
compare with an unscreened solve. Screening fixes only certified zeros and
preserves the complete penalty, including fusion edges touching removed zeros.
Returned coefficients keep their original feature order and dimension.

`dual_certificate()`, `dual_objective()`, and `duality_gap()` return optional
original-problem diagnostics. `screening_diagnostics()` reports certificate
passes, zero tests, certified features, remaining dimensions, and rebuilds. The
full original-loss PG mapping still controls convergence. At zero sparsity
strength, a feasible zero certificate can give a loose gap and screening leaves
features eligible. Fusion screening and new working sets remain deferred.

Certificates run initially, every ten accepted updates, and at return. Existing
proximal paths reset restrictions at every penalty point so features can
reactivate. Boxed runtime components retain screening capabilities; unsupported
wrappers and product-only designs continue fitting without screening.

API:
[`shrinkage::ProximalGradient`](https://shrinkage.cc/packages/rust/reference/dbf207e0ed64fdc5c0f5cc6c364e567527bca92737c69f45f608ed62e8d669d5).
