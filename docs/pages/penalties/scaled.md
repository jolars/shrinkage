# Scaled

`Scaled` multiplies an entire penalty, including its existing strengths:

$$
  P_{\mathrm{scaled}}(\beta) = aP(\beta),\qquad a \geq 0.
$$

For $a > 0$, its exact proximal map is the inner penalty's map with step $sa$.
For $a = 0$, it returns zero cost and the identity proximal map, while still
validating the inner parameters and dimensions.

## Constraint region

![The unit L1 diamond and the smaller diamond after doubling its
penalty.](figures/scaled.svg)

Top: `L1::new(1.0)`. Bottom: `Scaled::new(2.0, L1::new(1.0))`. At budget one,
doubling L1 halves each intercept of its diamond. A zero multiplier makes the
entire plane feasible. See the [plot
conventions](index.md#read-the-constraint-plots).

## Use the wrapper

```rust
use shrinkage::{L1, Penalty, ProximalPenalty, Scaled};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let penalty = Scaled::new(2.0, L1::new(0.5));
    assert_eq!(penalty.value(&[2.0, -1.0])?, 3.0);
    let mut shrunk = [0.0; 2];
    penalty.prox(&[2.0, -1.0], 1.0, &mut shrunk)?;
    assert_eq!(shrunk, [1.0, 0.0]);
    Ok(())
}
```

The multiplier must be finite and nonnegative. Scaling a value-only term does
not supply an exact proximal map. A scaled built-in proximal penalty supports
[proximal gradient](../reference/solvers/proximal-gradient.md), but the wrapper
does not add coordinate, IRLS, or hybrid compatibility. Compose it inside
[AxisPenalty](axis.md) for matrix coefficients.

API:
[`shrinkage::Scaled`](https://shrinkage.cc/packages/rust/reference/002e330ba78b1cd4f3d636338d545dc3876d2d58d4a1fda37a736ec8d9bb63e9).
