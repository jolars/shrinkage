# Masked

`Masked` excludes selected coefficients from a supported penalty. Let $m_j$ be
one for penalized coordinates and zero for excluded coordinates. Then

$$
  P_{\mathrm{masked}}(\beta) = P(m \odot \beta).
$$

In the Rust mask, `true` means **unpenalized**, while `false` retains the
penalty. The mask has one entry per coefficient and never includes intercepts.

## Constraint region

![A horizontal strip bounded at beta two equal to plus or minus one, continuing
in both horizontal directions.](figures/masked.svg)

`Masked::new(vec![true, false], L1::new(1.0))` gives $|\beta_2| \leq 1$. The
first coordinate is free, so the region is unbounded horizontally. See the [plot
conventions](index.md#read-the-constraint-plots).

## Use the wrapper

```rust
use shrinkage::{L1, Masked, Penalty, ProximalPenalty};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let penalty = Masked::new(vec![true, false], L1::new(0.5));
    assert_eq!(penalty.value(&[100.0, -2.0])?, 1.0);
    let mut shrunk = [0.0; 2];
    penalty.prox(&[100.0, -2.0], 1.0, &mut shrunk)?;
    assert_eq!(shrunk, [100.0, -1.5]);
    Ok(())
}
```

The proximal map copies excluded inputs unchanged and solves the restricted
penalty exactly. L1, ridge, and elastic net implement the required
`RestrictedProximalPenalty` capability. `Scaled` preserves it. Coupled penalties
cannot be masked merely by restoring coefficients after an unrestricted prox.

Use [proximal gradient](../reference/solvers/proximal-gradient.md) and compose
the wrapper inside [AxisPenalty](axis.md) when needed. The wrapper does not
supply the coordinate capability required by IRLS.

API:
[`shrinkage::Masked`](https://shrinkage.cc/packages/rust/reference/2dec36b360b55f50673aea5f43199a63294c9255ade7907fb4ade3d975bb106c)
and
[`shrinkage::RestrictedProximalPenalty`](https://shrinkage.cc/packages/rust/reference/6e81c84ae32917d40a6711abfffe4b252008f5265d347b3a8b6129238b41dffe).
