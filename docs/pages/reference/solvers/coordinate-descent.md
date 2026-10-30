# Coordinate descent

`CoordinateDescent` fits Gaussian L1 problems by cyclic updates of individual
coefficients, maintaining reusable residuals. It is also the solver behind the
`Lasso` convenience API. Each coordinate solves its exact one-dimensional
quadratic-plus-L1 problem.

For a feasible observation-scale dual point $\theta$, Gaussian lasso uses

$$
  D(\theta) = y^\mathsf{T}\theta - \frac{n}{2}\|\theta\|_2^2,
  \qquad
  \|\widetilde X^\mathsf{T}\theta\|_\infty \leq \lambda.
$$

A free intercept also requires $\mathbf{1}^\mathsf{T}\theta = 0$. The gap is
$G(\beta_0,\beta,\theta) = P(\beta_0,\beta) - D(\theta)$, using the [shared
primal objective convention](../../modeling.md#objective-and-notation).

## Fit a Gaussian lasso

```rust
use faer::Mat;
use shrinkage::{CoordinateDescent, Gaussian, L1, Problem, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let fit = Problem::new(&x, Gaussian::new(&[1.0, 3.0, 5.0]), L1::new(0.1))
        .fit_with(&CoordinateDescent::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Coefficients: {:?}", fit.coefficients());
    Ok(())
}
```

## Controls and convergence

The default budget is 10,000 coordinate sweeps. For positive strength, the
default relative duality-gap tolerance is `1e-6`; zero-strength problems instead
use absolute KKT tolerance `1e-6`. `.tolerance(value)` selects absolute KKT
stopping. `.terminate_on(...)` selects a supported criterion explicitly. At zero
strength, the solver's dual construction can yield a loose gap, so absolute KKT
stopping is the default.

Explicit and automatic regularization paths reuse preprocessing and residual
storage. Gap Safe screening is enabled for paths by default and controlled with
`.screening(...)`. Working sets are enabled for single fits and paths; see
[working sets](index.md#working-sets) for policy selection and diagnostics. A
sweep visits the current working set. Full checks run at least every ten sweeps
and before returning.

Dense and CSC input use column access and $O(n + p)$ solver storage. Sparse
coordinate updates operate on stored column entries. The solver requires
`RawColumns` and `ColumnStats`; it supports typed and runtime Gaussian L1
problems. Other datafits and penalties require a different solver.

API:
[`shrinkage::CoordinateDescent`](https://shrinkage.cc/packages/rust/reference/a483abd31a32ac227c08ded97cd2308397e21a231635317d7b09559f288d410d).
