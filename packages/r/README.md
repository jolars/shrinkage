# Shrinkage for R

Shrinkage fits Gaussian lasso regression with dense R matrices and sparse
`Matrix::dgCMatrix` matrices. It uses the Shrinkage Rust solver for single fits,
regularization paths, and cross-validation.

## Installation

Download the `shrinkage_<version>.tar.gz` archive from the `shrinkage-r-source`
artifact in a successful [CI
run](https://github.com/jolars/shrinkage/actions/workflows/ci.yml) and install
it from a terminal:

```sh
R CMD INSTALL shrinkage_<version>.tar.gz
```

Source builds require R 4.4 or later, Matrix, a C build toolchain, and Rust and
Cargo 1.89 or later. Cargo downloads locked dependencies. On Windows, install
Rtools and run `rustup target add x86_64-pc-windows-gnu`.

From a checkout, run `task r:install` to install into your R library or
`task r:build` to create a source tarball in `dist/`. Use `task r:check` to
build and check the tarball outside the checkout.

## Usage

```r
library(shrinkage)

x <- cbind(weight = mtcars$wt, power = mtcars$hp)
y <- mtcars$mpg

fit <- lasso(x, y, lambda = 0.5)
coef(fit)
predict(fit, x)
fit$diagnostics

path <- lasso(x, y)
coef(path)

cv <- cv_lasso(x, y, nfolds = 5, seed = 42, rule = "one_se")
cv$lambda[cv$selected_index]
predict(cv, x)
```

By default, the model fits an unpenalized intercept and standardizes the
predictors using population standard deviations. Coefficients and predictions
are on the original scale. See `?lasso` and `?cv_lasso` for options and return
values.
