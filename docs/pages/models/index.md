# Models

A statistical model combines a datafit and penalty. A solver supplies the
numerical method; it is selected explicitly rather than inferred from the
model's name.

  | Model                                      | Construction                              | Prediction                        |
  | ------------------------------------------ | ----------------------------------------- | --------------------------------- |
  | [Linear regression](linear.md)             | Gaussian with a compatible penalty        | Response mean                     |
  | [Logistic regression](logistic.md)         | Binomial with a compatible penalty        | Logit; transform to a probability |
  | [Multivariate regression](multivariate.md) | MultivariateGaussian with AxisPenalty     | Response-major means              |
  | [Lasso convenience API](lasso.md)          | Named Gaussian L1 configuration           | Response mean                     |
  | [SLOPE](slope.md)                          | Gaussian or binomial with Slope           | Response mean or logit            |
  | [Clustered lasso](clustered-lasso.md)      | Gaussian or binomial with ClusteredLasso  | Response mean or logit            |
  | [Chain fused lasso](chain-fused-lasso.md)  | Gaussian or binomial with ChainFusedLasso | Response mean or logit            |

Use `Problem::new(&x, datafit, penalty).fit_with(&solver)` for typed
composition. The type system checks supported combinations. Runtime problems
permit dynamic selection with explicit capability checks; see the [solver
compatibility table](../reference/solvers/index.md).

`Lasso` is currently the sole named convenience model. Other model families use
composition. The [shared conventions](../modeling.md) describe intercepts,
normalization, penalty scales, and original-scale output. The Python package
exposes Gaussian lasso and composition with Gaussian or binomial datafits, the
three MVP penalties, and explicit hybrid or proximal-gradient solvers.

API:
[`shrinkage::Problem`](https://shrinkage.cc/packages/rust/reference/4fd51b866367bd6673d82fc698be06078f421c181aef012c7bc5b92a97232647)
and
[`shrinkage::RuntimeProblem`](https://shrinkage.cc/packages/rust/reference/6c4676e8a2d70e42d12ccb31a1ad91f9cca07ac8509696f4c1996761f69e7999).
