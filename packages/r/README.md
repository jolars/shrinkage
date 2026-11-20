<!-- Generated from README.qmd with task r:readme. -->

# Shrinkage for R

Shrinkage fits Gaussian lasso regression with dense R matrices and sparse
`Matrix::dgCMatrix` matrices. It uses the Shrinkage Rust solver for single fits,
regularization paths, and cross-validation.

## Installation

Download and extract the `shrinkage-r-source` artifact from a successful [CI
run](https://github.com/jolars/shrinkage/actions/workflows/ci.yml). In R, run
the following and select the extracted `shrinkage_<version>.tar.gz` file when
prompted.

```r
pak::pak(paste0("local::", file.choose()))
```

Source builds require R 4.4 or later, a C build toolchain, and Rust and Cargo
1.89 or later. Cargo downloads locked dependencies. On Windows, install Rtools
and run `rustup target add x86_64-pc-windows-gnu`.

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
