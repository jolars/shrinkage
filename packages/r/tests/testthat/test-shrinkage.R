test_that("default and explicit Gaussian L1 specifications select coordinate descent", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$mpg
  default <- shrinkage(x, y)
  expect_s3_class(default, "shrinkage_path")
  expect_identical(default$fits[[1L]]$solver, "coordinate_descent")

  spec <- path_explicit(list(penalty_l1(0.1)))
  path <- shrinkage(x, y, penalty = spec)
  expect_s3_class(path, "shrinkage_path")
  expect_identical(path$penalties, spec$penalties)
  expect_equal(path$lambda, 0.1)
})

test_that("binomial fits and cross-validation retain their loss and penalties", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$am
  loss <- loss_binomial()
  penalties <- list(
    penalty_clustered_lasso(0.1, 0.1),
    penalty_clustered_lasso(0.1, 0.02)
  )
  cv <- shrinkage(
    x,
    y,
    loss,
    path_explicit(penalties),
    cv = cv_options(nfolds = 3)
  )
  expect_s3_class(cv, "shrinkage_cv")
  expect_identical(cv$path$penalties, penalties)
  expect_null(cv$path$lambda)
  expect_equal(predict(cv, x, type = "response"), plogis(predict(cv, x)))
  expect_equal(predict(cv, x), predict(cv$path, x)[, cv$selected_index])

  sparse <- methods::as(Matrix::Matrix(x, sparse = TRUE), "CsparseMatrix")
  probit <- shrinkage(sparse, y, loss_binomial("probit"), penalty_ridge(0.1))
  expect_equal(predict(probit, x, type = "response"), pnorm(predict(probit, x)))
  expect_error(
    shrinkage(x, y, loss, penalties[[1L]], solver = "coordinate_descent"),
    "solver.*coordinate_descent"
  )
})

test_that("automatic paths retain complete elastic net penalties", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$mpg
  path <- shrinkage(
    x,
    y,
    penalty = path_auto(
      penalty_elastic_net(1, 0.5),
      nlambda = 4,
      early_stopping = FALSE
    )
  )
  expect_length(path$fits, 4L)
  expect_true(all(diff(path$lambda) < 0))
  for (i in seq_along(path$fits)) {
    pen <- path$penalties[[i]]
    expect_equal(pen$l2, 0.5 * pen$l1)
    expect_equal(
      coef(path$fits[[i]]),
      coef(shrinkage(x, y, penalty = pen)),
      tolerance = 1e-4
    )
  }
})

test_that("mixed paths support distinct proximal penalties", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$mpg
  penalties <- list(
    penalty_group_lasso(0.5, list(1:2)),
    penalty_scaled(penalty_masked(penalty_l1(0.5), c(TRUE, FALSE)), scale = 2),
    penalty_chain_fused_lasso(0.1, 0.02, weights = 1),
    penalty_slope(c(0.2, 0.1))
  )
  path <- shrinkage(x, y, penalty = path_explicit(penalties))
  expect_identical(path$penalties, penalties)
  expect_null(path$lambda)
  for (i in seq_along(penalties)) {
    fit <- shrinkage(
      x,
      y,
      penalty = penalties[[i]],
      solver = "proximal_gradient"
    )
    expect_equal(coef(path)[, i], coef(fit), tolerance = 1e-4)
  }

  cv <- shrinkage(
    x,
    y,
    penalty = path_explicit(penalties),
    cv = cv_options(nfolds = 3)
  )
  expect_identical(cv$path$penalties, penalties)
  expect_equal(cv$mean_scores, colMeans(cv$fold_scores))
})

test_that("multivariate paths retain coefficient axes and response shapes", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- cbind(mpg = mtcars$mpg, gear = mtcars$gear)
  loss <- loss_multivariate_gaussian()
  penalty <- penalty_axis(penalty_ridge(0.1), "responses")
  fit <- shrinkage(x, y, loss, penalty)
  expect_identical(fit$penalty, penalty)
  expect_equal(dim(coef(fit)), c(3L, 2L))
  expect_equal(colnames(coef(fit)), colnames(y))
  expect_equal(predict(fit, x), cbind(1, x) %*% coef(fit))

  path <- shrinkage(x, y, loss, path_auto(penalty, nlambda = 3))
  expect_equal(dim(coef(path)), c(3L, 2L, 3L))
  expect_equal(dim(predict(path, x)), c(nrow(x), 2L, 3L))
  expect_equal(predict(path, x)[,, 2L], predict(path$fits[[2L]], x))
  expect_error(shrinkage(x, y, loss, cv = cv_options()), "multivariate")
})
