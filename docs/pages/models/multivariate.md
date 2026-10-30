# Multivariate regression

A multivariate Gaussian model estimates $q$ responses from the same design:

$$
  \min_{\beta_0,\beta}\;\frac{\|Y - \mathbf{1}\beta_0^\mathsf{T} - \widetilde X\beta\|_F^2}{2n}
  + R_{\mathrm{axis}}(\beta).
$$

Here $\beta$ is a $p \times q$ coefficient matrix, and $\beta_0$ is a vector of
$q$ intercepts. Use an explicit $p$-feature, $q$-response coefficient layout and
choose which vectors the penalty sees. The loss sums over responses, so adding
responses does not divide either the loss or the penalty strengths by $q$.

## Fit two responses

```rust
use faer::{Col, Mat};
use shrinkage::{
    AxisPenalty, CoefficientLayout, MultivariateGaussian, PenaltyAxis, Problem, ProximalGradient,
    Ridge, Termination,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(4, 2, |i, j| {
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]][i][j]
    });
    let y = [1.0, 5.0, 4.0, 2.0, -6.0, 2.0, -3.0, -1.0];
    let layout = CoefficientLayout::new(2, 2)?;
    let penalty = AxisPenalty::new(layout, PenaltyAxis::Features, Ridge::new(0.5));
    let fit = Problem::new(&x, MultivariateGaussian::new(&y, 4, 2), penalty)
        .fit_with(&ProximalGradient::<Col<f64>>::new())?;
    assert_eq!(fit.termination(), Termination::Converged);
    println!("First response: {:?}", fit.coefficients_for_response(0));
    println!("Intercepts: {:?}", fit.intercepts());
    println!("Response-major predictions: {:?}", fit.predict(&x)?);
    Ok(())
}
```

Observations, coefficients, and predictions store each response column
contiguously. The model shares feature preprocessing and fits a separate
unpenalized intercept per response.

`PenaltyAxis::Features` applies the vector penalty independently to every
response. To select a feature jointly across responses, use
`PenaltyAxis::Responses` with an inner Group LASSO containing one group covering
all response coordinates. `PenaltyAxis::All` applies one complete penalty to the
flattened coefficient matrix. See [AxisPenalty](../penalties/axis.md).

Only [proximal gradient](../reference/solvers/proximal-gradient.md) currently
fits this layout. Typed and runtime multivariate problems share the same
iteration.
