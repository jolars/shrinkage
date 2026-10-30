# AxisPenalty

`AxisPenalty` applies a complete vector penalty along explicitly selected axes
of a $p$-feature, $q$-response coefficient matrix $\beta$:

$$
  \begin{aligned}
    P_{\mathrm{Features}}(\beta)  & = \sum_{r=1}^q P(\beta_{:,r}),  \\
    P_{\mathrm{Responses}}(\beta) & = \sum_{j=1}^p P(\beta_{j,:}),  \\
    P_{\mathrm{All}}(\beta)       & = P(\operatorname{vec}(\beta)).
  \end{aligned}
$$

Each group receives the supplied strength. Shrinkage does not divide by the
number of features, responses, or groups. Intercepts remain separate.
`CoefficientLayout` stores response columns contiguously, with index
`response * nfeatures + feature`.

## Constraint region

![A diamond for separate scalar penalties and a disk for one joint response
penalty.](figures/axis.svg)

Both panels use one feature and two responses, with the unsquared L2 norm as the
inner penalty, represented by one Group LASSO group of unit weight and strength
one at each required vector width. The axes denote the two response coefficients
of that feature. `Features` gives $|\beta_1| + |\beta_2| \leq 1$; `Responses`
gives $\sqrt{\beta_1^2 + \beta_2^2} \leq 1$. `All` coincides with `Responses` in
this layout. See the [plot conventions](index.md#read-the-constraint-plots).

## Use the wrapper

```rust
use shrinkage::{AxisPenalty, CoefficientLayout, Penalty, PenaltyAxis, Ridge};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layout = CoefficientLayout::new(2, 2)?;
    let penalty = AxisPenalty::new(layout, PenaltyAxis::Features, Ridge::new(1.0));
    let coefficients = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(penalty.value(&coefficients)?, 15.0);
    Ok(())
}
```

Use `Features` to fit responses independently, `Responses` to couple responses
within a feature, and `All` to apply one penalty to the flattened matrix.
Disjoint groups make the composed proximal map exact whenever the inner map is
exact. `Scaled` and `Masked` can wrap the inner penalty; mask length follows the
selected vector width.

Fitting uses [proximal gradient](../reference/solvers/proximal-gradient.md) with
[multivariate Gaussian](../datafits/multivariate-gaussian.md).

API:
[`shrinkage::AxisPenalty`](https://shrinkage.cc/packages/rust/reference/08140b02568bb11bf1bb6237342dadbfb2c025ac93d35b1a67ee6c91fa412ca3),
[`shrinkage::PenaltyAxis`](https://shrinkage.cc/packages/rust/reference/df36239cb375fb6962f88368bd5b46029efaae9ee19482d2d02a669717981971),
and
[`shrinkage::CoefficientLayout`](https://shrinkage.cc/packages/rust/reference/62149afd3598115cd84b1d082cb6c3684abff0b85239710acbface17a12565fe).
