# IRLS

`Irls` fits Gaussian and binomial datafits with separable convex penalties. It
constructs a quadratic approximation from the predictor gradient and diagonal
Hessian, then uses weighted coordinate descent for the inner solve. Backtracking
evaluates the original penalized objective.

L1, ridge, and elastic net provide the required `CoordinatePenalty` capability.
Coupled penalties such as SLOPE or fusion instead use proximal gradient or the
hybrid solver.

## Fit a binomial lasso

```rust
use faer::Mat;
use shrinkage::{Binomial, Irls, L1, Problem, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(4, 1, |i, _| i as f64);
    let y = [0.0, 0.0, 1.0, 1.0];
    let fit = Problem::new(&x, Binomial::new(&y), L1::new(0.1)).fit_with(&Irls::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("Logits: {:?}", fit.predict(&x)?);
    Ok(())
}
```

## Controls and convergence

Defaults are 100 accepted outer steps, 1,000 coordinate sweeps per quadratic
model, at most 100 outer step halvings, working sets enabled, and absolute KKT
tolerance `1e-6`. Configure `.max_iterations(...)`,
`.max_inner_iterations(...)`, `.max_backtracks(...)`,
`.working_set_policy(...)`, and `.tolerance(...)`.

The default [working-set policy](index.md#working-sets) uses adaptive sets for
single fits and ever-active sets for paths. Scores refresh whenever the
quadratic model changes. Restricted checks run between complete quadratic
checks, with a full check at least every ten sweeps and before accepting an
inner solution.

Only a complete KKT check on the original loss certifies convergence. An inexact
inner solve can still provide a descent direction; meeting an inner budget does
not establish outer convergence. The quadratic model floors predictor curvature
at $10^{-12}/n$ while leaving the original loss, gradient, and final KKT check
unchanged. IRLS provides no dual certificate or safe screening.

The solver uses typed column-access problems and $O(n + p)$ workspace. Explicit
and supported automatic paths reuse that workspace. Binomial predictions are
logits; see [logistic regression](../../models/logistic.md) for probabilities.

API:
[`shrinkage::Irls`](https://shrinkage.cc/packages/rust/reference/38d3c0c77c3a369dc72fa82940fe75e00e04888f78613113e69bdc12b1c8d7dd)
and
[`shrinkage::IrlsFit`](https://shrinkage.cc/packages/rust/reference/20c675c8b6635b5652c30df90b363678987705f6c4e43dc91353d9331c3056c4).
