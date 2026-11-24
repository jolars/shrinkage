test_that("single-feature fits match the closed-form solution", {
  x <- cbind(feature = 0:2)
  y <- c(1L, 3L, 5L)
  sparse <- methods::as(Matrix::Matrix(x, sparse = TRUE), "CsparseMatrix")
  fit <- shrinkage(
    x,
    y,
    penalty = penalty_l1(0.1),
    normalization = "none",
    tolerance = 1e-10
  )
  expect_equal(
    coef(fit),
    c("(Intercept)" = 1.15, feature = 1.85),
    tolerance = 1e-6
  )
  expect_equal(predict(fit, x), c(1.15, 3, 4.85), tolerance = 1e-6)
  sparse_fit <- shrinkage(
    sparse,
    y,
    penalty = penalty_l1(0.1),
    normalization = "none",
    tolerance = 1e-10
  )
  expect_equal(coef(sparse_fit), coef(fit), tolerance = 1e-6)
  expect_equal(predict(fit, sparse), predict(fit, x), tolerance = 1e-6)
  expect_equal(
    coef(shrinkage(x, y, penalty = penalty_l1(0))),
    c("(Intercept)" = 1, feature = 2),
    tolerance = 1e-6
  )
  empty <- matrix(numeric(), nrow = 3L, ncol = 0L)
  expect_equal(
    predict(shrinkage(empty, y, penalty = penalty_l1(1)), empty),
    rep(3, 3),
    tolerance = 1e-6
  )
  expect_length(predict(fit, x[FALSE, , drop = FALSE]), 0L)
})

test_that("lasso matches the Rust-generated fixture shared with Python", {
  fixture_x <- cbind(
    0:7,
    c(1, 0, 2, -1, 3, 1, 4, -2),
    3,
    c(0, 2, 0, 1, 0, 2, 1, 0)
  )
  fixture_y <- c(1, 2.3, 0.8, 3.9, 4.2, 2.7, 6.1, 5.4)
  fixture <- shrinkage(fixture_x, fixture_y, penalty = penalty_l1(0.12))
  expect_equal(
    unname(coef(fixture)),
    c(1.158303027798234, 0.611913420629076, 0, 0, 0),
    tolerance = 1e-6
  )
  expect_equal(
    predict(fixture, rbind(c(1.5, -0.5, 3, 1), c(8, 2, 3, 0))),
    c(2.076173158741848, 6.0536103928308425),
    tolerance = 1e-6
  )
})

test_that("lasso paths preserve supplied strengths and predictions", {
  x <- cbind(feature = 0:2)
  y <- c(1L, 3L, 5L)
  sparse <- methods::as(Matrix::Matrix(x, sparse = TRUE), "CsparseMatrix")
  strengths <- c(0.3, 0.1, 0.3, 0)
  candidates <- path_explicit(lapply(strengths, penalty_l1))
  path <- shrinkage(x, y, penalty = candidates)
  expect_equal(path$lambda, strengths, tolerance = 1e-6)
  expect_equal(
    coef(path),
    coef(shrinkage(sparse, y, penalty = candidates)),
    tolerance = 1e-6
  )
  expect_equal(predict(path, sparse), predict(path, x), tolerance = 1e-6)
  expect_equal(
    predict(unserialize(serialize(path, NULL)), x),
    predict(path, x),
    tolerance = 1e-6
  )
})

test_that("automatic paths support early stopping", {
  x <- cbind(feature = 0:2)
  y <- c(1L, 3L, 5L)
  automatic <- shrinkage(x, y)
  expect_lt(length(automatic$lambda), 100L)
  expect_false(automatic$termination == "completed")
  expect_length(
    shrinkage(
      x,
      y,
      penalty = path_auto(nlambda = 7, early_stopping = FALSE)
    )$lambda,
    7L
  )
})

test_that("cross-validation scores held-out folds and selects a model", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$mpg
  sparse <- methods::as(Matrix::Matrix(x, sparse = TRUE), "CsparseMatrix")
  ids <- rep_len(1:3, nrow(x))
  cv <- shrinkage(
    x,
    y,
    penalty = path_auto(nlambda = 8),
    cv = cv_options(foldid = ids)
  )
  cv_sparse <- shrinkage(
    sparse,
    y,
    penalty = path_auto(nlambda = 8),
    cv = cv_options(foldid = ids, rule = "one_se")
  )
  expect_equal(cv$fold_scores, cv_sparse$fold_scores, tolerance = 1e-6)
  expect_equal(cv$mean_scores, colMeans(cv$fold_scores), tolerance = 1e-6)
  expect_equal(
    cv$standard_errors,
    apply(cv$fold_scores, 2L, sd) / sqrt(3),
    tolerance = 1e-6
  )
  expect_identical(cv$foldid, ids)
  expect_equal(cv$selected_index, which.min(cv$mean_scores))
  expect_equal(cv_sparse$selected_index, cv_sparse$one_se_index)
  min <- cv$minimum_index
  threshold <- cv$mean_scores[min] + cv$standard_errors[min]
  expect_equal(cv$one_se_index, which(cv$mean_scores <= threshold)[1L])
  for (i in 1:3) {
    train <- ids != i
    training_path <- shrinkage(
      x[train, ],
      y[train],
      penalty = path_auto(nlambda = 8, early_stopping = FALSE)
    )
    expect_equal(cv$fold_lambdas[i, ], training_path$lambda, tolerance = 1e-6)
    errors <- predict(training_path, x[!train, ]) - y[!train]
    expect_equal(cv$fold_scores[i, ], colMeans(errors^2), tolerance = 1e-6)
  }
  expect_equal(
    predict(cv_sparse, sparse),
    predict(cv_sparse$path, sparse)[, cv_sparse$selected_index],
    tolerance = 1e-6
  )
})

test_that("seeded cross-validation repeats fold assignments", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$mpg
  strengths <- c(0.3, 0.1, 0.3, 0)
  candidates <- path_explicit(lapply(strengths, penalty_l1))
  cv_a <- shrinkage(
    x,
    y,
    penalty = candidates,
    cv = cv_options(nfolds = 4, seed = 42)
  )
  cv_b <- shrinkage(
    x,
    y,
    penalty = candidates,
    cv = cv_options(nfolds = 4, seed = 42)
  )
  expect_equal(cv_a$foldid, cv_b$foldid, tolerance = 1e-6)
  expect_equal(cv_a$lambda, strengths, tolerance = 1e-6)
})

test_that("iteration limits are reported for fits and cross-validation", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$mpg
  limited <- shrinkage(
    x,
    y,
    penalty = penalty_l1(0.01),
    tolerance = 1e-15,
    max_iterations = 1
  )
  expect_equal(limited$diagnostics$termination, "iteration_limit")
  expect_error(
    shrinkage(
      x,
      y,
      penalty = penalty_l1(0.01),
      cv = cv_options(nfolds = 3),
      tolerance = 1e-15,
      max_iterations = 1
    ),
    "converg"
  )
})

test_that("lasso rejects invalid inputs and prediction dimensions", {
  x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
  y <- mtcars$mpg
  sparse <- methods::as(Matrix::Matrix(x, sparse = TRUE), "CsparseMatrix")
  expect_error(shrinkage(x, y[-1]), "one value")
  expect_error(shrinkage(x, y, penalty = penalty_l1(Inf)), "lambda")
  expect_error(shrinkage(x, replace(y, 1L, NA)), "finite")
  expect_error(
    shrinkage(x, y, cv = cv_options(foldid = rep_len(c(1, 3), length(y)))),
    "fold"
  )
  fit <- shrinkage(
    x,
    y,
    penalty = penalty_l1(0.01),
    tolerance = 1e-15,
    max_iterations = 1
  )
  expect_error(predict(fit, x[, 1L, drop = FALSE]), "same number")
  bad_sparse <- sparse
  bad_sparse@i[1L] <- as.integer(nrow(x))
  expect_error(shrinkage(bad_sparse, y), "invalid class")
})
