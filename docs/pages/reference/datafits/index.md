# Datafits

A datafit defines the loss on the complete linear predictor, including the
intercept. Matrix operations belong to the design. Loss values and derivatives
use the same averaging convention.

  | Datafit                                           | Response                                 | Loss                                                                | Compatible solvers                                                                 |
  | ------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
  | [Gaussian](gaussian.md)                           | One finite real response per observation | Averaged squared error                                              | Coordinate descent for L1, proximal gradient, IRLS, hybrid for supported penalties |
  | [Binomial](binomial.md)                           | Finite values in $[0,1]$                 | Averaged Bernoulli negative log likelihood                          | Proximal gradient, IRLS, hybrid for supported penalties                            |
  | [Multivariate Gaussian](multivariate-gaussian.md) | Multiple finite response columns         | Squared error, summed over responses and averaged over observations | Proximal gradient with AxisPenalty                                                 |

All three fit unpenalized intercepts by default. Response dimensions must match
nonempty training observations. Data and parameters are validated when losses
are evaluated or models are fitted.

[`shrinkage::Datafit`](https://shrinkage.cc/packages/rust/reference/9d16367f3210bbbb9bf039290cf9c94f447e51d4ea33aea446a7f30e965d873a)
supplies loss evaluation,
[`shrinkage::SmoothDatafit`](https://shrinkage.cc/packages/rust/reference/f6f2ab3078b0ea56f174e2981b999f9be72be3c00ca00ad6485be1ed341db8c1)
adds gradients, and
[`shrinkage::DiagonalHessianDatafit`](https://shrinkage.cc/packages/rust/reference/0c7a483f4e622180bfcc5efab524b3377e2104e91e0d263eb5a204d4a4ab722e)
supplies the predictor curvature used by IRLS. These capabilities determine
which [solvers](../solvers/index.md) can fit a composition.
