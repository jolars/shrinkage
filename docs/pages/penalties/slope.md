# SLOPE

SLOPE assigns larger weights to larger coefficient magnitudes. Here
$|\beta|_{(j)}$ denotes the $j$th largest absolute value. Its corners can
encourage equal magnitudes, including coefficients with opposite signs. Supply
exactly one finite, nonnegative, nonincreasing weight per coefficient; the
weights already include the penalty strength.

$$
  R(\beta) = \sum_{j=1}^p w_j|\beta|_{(j)},\qquad w_1
           \geq \cdots \geq w_p \geq 0.
$$

## Constraint region

![Feasible region for slope; the shaded set includes its
boundary.](figures/slope.svg)

$w = (1,0.5)$:
$\max(|\beta_1|,|\beta_2|) + 0.5\min(|\beta_1|,|\beta_2|) \leq 1$. See the [plot
conventions](index.md#read-the-constraint-plots) for the fixed budget and
coefficient scale.

## Use the penalty

```rust
use shrinkage::{Penalty, ProximalPenalty, Slope};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let penalty = Slope::new(vec![0.5, 0.25]);
    let coefficients = [2.0, -1.0];
    println!("Penalty: {}", penalty.value(&coefficients)?);
    let mut shrunk = [0.0; 2];
    penalty.prox(&coefficients, 1.0, &mut shrunk)?;
    println!("Proximal result: {shrunk:?}");
    Ok(())
}
```

Equal weights recover lasso. Repeated weights and trailing zeros are valid;
all-zero weights give no penalty. Use [proximal
gradient](../reference/solvers/proximal-gradient.md) or the [hybrid
solver](../reference/solvers/hybrid.md) with Gaussian or binomial datafits. The
[SLOPE fitting guide](../models/slope.md) explains magnitude clusters and
advanced controls.

API:
[`shrinkage::Slope`](https://shrinkage.cc/packages/rust/reference/b160a8286206ec7afad932df7849dd0eb34050fc3d7a58465f4671287121ba7c).
