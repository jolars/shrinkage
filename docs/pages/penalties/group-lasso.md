# Group LASSO

Group LASSO selects explicitly defined groups of coefficients together:

$$
  P(\beta) = \lambda\sum_{g=1}^G w_g\|\beta_g\|_2.
$$

Groups must be disjoint, nonempty, and contain valid coefficient indices.
`DisjointGroups` checks these requirements at construction. Supply one finite,
nonnegative weight per group. Shrinkage applies those weights directly, without
implicit group-size scaling. For a size adjustment, supply weights such as
$w_g = \sqrt{|g|}$ explicitly.

## Constraint region

![A disk for one two-coordinate group and a diamond for two singleton
groups.](figures/group-lasso.svg)

Both panels use $\lambda = 1$ and unit weights. One group $\{1,2\}$ gives the
unsquared L2 constraint $\sqrt{\beta_1^2 + \beta_2^2} \leq 1$. Two singleton
groups give $|\beta_1| + |\beta_2| \leq 1$. The circle coincides with a ridge
sublevel set in this example, although the penalty values and shrinkage differ.
See the [plot conventions](index.md#read-the-constraint-plots).

## Use the penalty

```rust
use shrinkage::{DisjointGroups, GroupLasso, Penalty, ProximalPenalty, UncoveredPolicy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let groups = DisjointGroups::new(3, vec![vec![0, 2]], vec![1.0], UncoveredPolicy::Unpenalized)?;
    let penalty = GroupLasso::new(0.5, groups);
    assert_eq!(penalty.value(&[3.0, 9.0, 4.0])?, 2.5);
    let mut shrunk = [0.0; 3];
    penalty.prox(&[3.0, 9.0, 4.0], 1.0, &mut shrunk)?;
    assert_eq!(shrunk[1], 9.0);
    Ok(())
}
```

`UncoveredPolicy::Reject` requires complete coverage.
`UncoveredPolicy::Unpenalized` leaves uncovered coefficients free. Zero-weight
groups are also unpenalized. A group can become exactly zero because the
proximal map shrinks its input vector by $\max(1 - s\lambda w_g/\|v_g\|_2,0)$,
with zero input mapped to zero.

Use [proximal gradient](../reference/solvers/proximal-gradient.md) with Gaussian
or binomial datafits. The [multivariate model](../models/multivariate.md) shows
how an axis penalty can select features across responses.

API:
[`shrinkage::GroupLasso`](https://shrinkage.cc/packages/rust/reference/0e1357425403f686e1e3ace6b0f0af21e368cafd2bcaef0932eafd1ce2a4cf9c)
and
[`shrinkage::DisjointGroups`](https://shrinkage.cc/packages/rust/reference/66837f98e8b085ca419007cb1086d289a594d7573fda3cdd4d56b38563de68bc).
