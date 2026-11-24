<!-- Generated from README.qmd with task r:readme. -->

# Shrinkage for R

Fit regularized models with `shrinkage()`. Choose a loss and a penalty to fit
one model, a regularization path, or a cross-validated path. The package accepts
dense numeric matrices and sparse `Matrix::dgCMatrix` matrices.

## Installation

From a checkout, run `task r:install`. You can also download the
`shrinkage-r-source` artifact from a successful [CI
run](https://github.com/jolars/shrinkage/actions/workflows/ci.yml) and install
the source archive with `R CMD INSTALL`.

Source builds require R 4.4 or later, a C build toolchain, and Rust and Cargo
1.89 or later. Cargo downloads locked dependencies. On Windows, install Rtools
and run `rustup target add x86_64-pc-windows-gnu`.

## Usage

`shrinkage(x, y)` fits an automatic Gaussian L1 path. Supply a complete penalty
to fit at one strength.

```r
library(shrinkage)

x <- cbind(weight = mtcars$wt, power = mtcars$hp)
fit <- shrinkage(x, mtcars$mpg, penalty = penalty_l1(0.5))
coef(fit)
predict(fit, x)
```

Choose another loss and penalty to fit a different model.

```r
logistic <- shrinkage(
  x,
  mtcars$am,
  loss = loss_binomial(),
  penalty = penalty_clustered_lasso(l1 = 0.1, fusion = 0.05)
)
predict(logistic, x, type = "response")
```

Use `path_auto()` to vary a scalar strength while preserving the penalty shape.
Use `path_explicit()` to supply complete penalties, for example to vary the two
elastic-net strengths independently.

```r
cv <- shrinkage(
  x,
  mtcars$mpg,
  penalty = path_auto(penalty_elastic_net(l1 = 1, l2 = 0.5), nlambda = 20),
  cv = cv_options(nfolds = 5, seed = 42)
)
cv$path$penalties[[cv$selected_index]]
predict(cv, x)
```

Models fit an unpenalized intercept and standardize predictors by default.
Coefficients and predictions use the original scale. See `?shrinkage` for
supported losses, solvers, and return values.
