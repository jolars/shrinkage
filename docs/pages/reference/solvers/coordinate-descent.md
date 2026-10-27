# Coordinate descent

`CoordinateDescent` fits Gaussian L1 problems by cyclic updates of individual
coefficients, maintaining reusable residuals. It is also the solver behind the
`Lasso` convenience API. Each coordinate solves its exact one-dimensional
quadratic-plus-L1 problem.

## Fit a Gaussian lasso

```rust
use faer::Mat;
use shrinkage::{CoordinateDescent, Gaussian, Problem, Termination, L1};

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

The default budget is 10,000 complete coordinate sweeps. For positive strength,
the default relative duality-gap tolerance is `1e-6`; zero-strength problems
instead use absolute KKT tolerance `1e-6`. `.tolerance(value)` selects absolute
KKT stopping. `.terminate_on(...)` selects a supported criterion explicitly. At
zero strength, the solver's dual construction can yield a loose gap, so absolute
KKT stopping is the default.

Explicit and automatic regularization paths reuse preprocessing and residual
storage. Gap Safe screening and working sets are enabled for paths by default;
single fits retain complete cyclic sweeps. Control them with `.screening(...)`
and `.working_sets(...)`.

Dense and CSC input use column access and $O(n + p)$ solver storage. Sparse
coordinate updates operate on stored column entries. The solver requires
`RawColumns` and `ColumnStats`; it supports typed and runtime Gaussian L1
problems. Other datafits and penalties require a different solver.

API:
[`shrinkage::CoordinateDescent`](https://shrinkage.cc/packages/rust/reference/a483abd31a32ac227c08ded97cd2308397e21a231635317d7b09559f288d410d).
