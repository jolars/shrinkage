library(shrinkage)

equal <- function(x, y, tolerance = 1e-6) {
  difference <- all.equal(x, y, tolerance = tolerance)
  if (!isTRUE(difference)) stop(paste(difference, collapse = "\n"))
}
fails <- function(expr, pattern) {
  error <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error)))
}

# The unstandardized single-feature objective has a closed-form solution.
x <- cbind(feature = 0:2)
y <- c(1L, 3L, 5L)
sparse <- methods::as(Matrix::Matrix(x, sparse = TRUE), "CsparseMatrix")
fit <- lasso(x, y, 0.1, standardize = FALSE, tolerance = 1e-10)
equal(coef(fit), c("(Intercept)" = 1.15, feature = 1.85))
equal(predict(fit, x), c(1.15, 3, 4.85))
equal(coef(lasso(sparse, y, 0.1, standardize = FALSE, tolerance = 1e-10)), coef(fit))
equal(predict(fit, sparse), predict(fit, x))
equal(coef(lasso(x, y, 0)), c("(Intercept)" = 1, feature = 2))
empty <- matrix(numeric(), nrow = 3L, ncol = 0L)
equal(predict(lasso(empty, y, 1), empty), rep(3, 3))
stopifnot(length(predict(fit, x[FALSE, , drop = FALSE])) == 0L)

# Compare with the Rust-generated lasso fixture also used by the Python package.
fixture_x <- cbind(0:7, c(1, 0, 2, -1, 3, 1, 4, -2), 3, c(0, 2, 0, 1, 0, 2, 1, 0))
fixture_y <- c(1, 2.3, 0.8, 3.9, 4.2, 2.7, 6.1, 5.4)
fixture <- lasso(fixture_x, fixture_y, 0.12)
equal(unname(coef(fixture)), c(1.158303027798234, 0.611913420629076, 0, 0, 0))
equal(predict(fixture, rbind(c(1.5, -0.5, 3, 1), c(8, 2, 3, 0))),
      c(2.076173158741848, 6.0536103928308425))

strengths <- c(0.3, 0.1, 0.3, 0)
path <- lasso(x, y, strengths)
equal(path$lambda, strengths)
equal(coef(path), coef(lasso(sparse, y, strengths)))
equal(predict(path, sparse), predict(path, x))
equal(predict(unserialize(serialize(path, NULL)), x), predict(path, x))
automatic <- lasso(x, y)
stopifnot(length(automatic$lambda) < 100L, automatic$termination != "completed")
stopifnot(length(lasso(x, y, nlambda = 7, early_stopping = FALSE)$lambda) == 7L)

x <- cbind(wt = mtcars$wt, hp = mtcars$hp)
y <- mtcars$mpg
sparse <- methods::as(Matrix::Matrix(x, sparse = TRUE), "CsparseMatrix")
ids <- rep_len(1:3, nrow(x))
cv <- cv_lasso(x, y, foldid = ids, nlambda = 8)
cv_sparse <- cv_lasso(sparse, y, foldid = ids, nlambda = 8, rule = "one_se")
equal(cv$fold_scores, cv_sparse$fold_scores)
equal(cv$mean_scores, colMeans(cv$fold_scores))
equal(cv$standard_errors, apply(cv$fold_scores, 2L, sd) / sqrt(3))
stopifnot(identical(cv$foldid, ids), cv$selected_index == which.min(cv$mean_scores),
          cv_sparse$selected_index == cv_sparse$one_se_index)
min <- cv$minimum_index
stopifnot(cv$one_se_index == which(cv$mean_scores <= cv$mean_scores[min] +
                                   cv$standard_errors[min])[1L])
for (i in 1:3) {
  train <- ids != i
  training_path <- lasso(x[train, ], y[train], nlambda = 8, early_stopping = FALSE)
  equal(cv$fold_lambdas[i, ], training_path$lambda)
  equal(cv$fold_scores[i, ], colMeans((predict(training_path, x[!train, ]) - y[!train])^2))
}
equal(predict(cv_sparse, sparse),
      predict(cv_sparse$path, sparse)[, cv_sparse$selected_index])
cv_a <- cv_lasso(x, y, strengths, nfolds = 4, seed = 42)
cv_b <- cv_lasso(x, y, strengths, nfolds = 4, seed = 42)
equal(cv_a$foldid, cv_b$foldid)
equal(cv_a$lambda, strengths)
limited <- lasso(x, y, 0.01, tolerance = 1e-15, max_iterations = 1)
stopifnot(limited$diagnostics$termination == "iteration_limit")
fails(cv_lasso(x, y, 0.01, nfolds = 3, tolerance = 1e-15, max_iterations = 1),
      "converg")

fails(lasso(x, y[-1], 1), "one value")
fails(lasso(x, y, c(1, Inf)), "lambda")
fails(lasso(x, replace(y, 1L, NA), 1), "finite")
fails(cv_lasso(x, y, 1, foldid = rep_len(c(1, 3), length(y))), "fold")
fails(predict(limited, x[, 1L, drop = FALSE]), "same number")
bad_sparse <- sparse
bad_sparse@i[1L] <- as.integer(nrow(x))
fails(lasso(bad_sparse, y, 1), "invalid class")
