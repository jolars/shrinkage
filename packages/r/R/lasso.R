numeric_values <- function(x) {
  is.numeric(x) && typeof(x) %in% c("double", "integer") && all(is.finite(x))
}

flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(name, " must be TRUE or FALSE", call. = FALSE)
  }
  x
}

count <- function(x, name, minimum = 1L) {
  if (!numeric_values(x) || length(x) != 1L ||
      x < minimum || x > .Machine$integer.max || x != floor(x)) {
    stop(name, " must be an integer >= ", minimum, call. = FALSE)
  }
  as.integer(x)
}

design <- function(x) {
  if (is(x, "dgCMatrix")) {
    validObject(x)
    values <- slot(x, "x")
    if (!all(is.finite(values))) stop("x must be finite", call. = FALSE)
    return(list(dim = dim(x), values = values, offsets = slot(x, "p"),
                indices = slot(x, "i")))
  }
  if (!is.matrix(x) || !numeric_values(x)) {
    stop("x must be a finite numeric matrix or dgCMatrix", call. = FALSE)
  }
  list(dim = dim(x), values = as.double(x), offsets = NULL)
}

fit_inputs <- function(x, y, lambda, intercept, standardize, tolerance,
                       max_iterations, nlambda, lambda_min_ratio, early_stopping) {
  matrix <- design(x)
  if (!numeric_values(y) || !is.null(dim(y)) ||
      length(y) != nrow(x) || !length(y)) {
    stop("y must be a finite vector with one value per observation", call. = FALSE)
  }
  if (!is.null(lambda) && (!numeric_values(lambda) || !is.null(dim(lambda)) ||
                          !length(lambda) || any(lambda < 0))) {
    stop("lambda must contain finite nonnegative strengths", call. = FALSE)
  }
  if (!is.null(tolerance) &&
      (!numeric_values(tolerance) || length(tolerance) != 1L || tolerance <= 0)) {
    stop("tolerance must be positive and finite", call. = FALSE)
  }
  if (is.null(lambda) && !is.null(lambda_min_ratio) &&
      (!numeric_values(lambda_min_ratio) || length(lambda_min_ratio) != 1L ||
       lambda_min_ratio <= 0 || lambda_min_ratio > 1)) {
    stop("lambda_min_ratio must be in (0, 1]", call. = FALSE)
  }
  list(
    design = matrix, y = as.double(y),
    lambda = if (!is.null(lambda)) as.double(lambda),
    options = list(
      intercept = flag(intercept, "intercept"),
      standardize = flag(standardize, "standardize"),
      tolerance = if (!is.null(tolerance)) as.double(tolerance),
      max_iterations = count(max_iterations, "max_iterations"),
      nlambda = if (is.null(lambda)) count(nlambda, "nlambda") else 1L,
      lambda_min_ratio = if (is.null(lambda) && !is.null(lambda_min_ratio))
        as.double(lambda_min_ratio),
      early_stopping = if (is.null(lambda)) flag(early_stopping, "early_stopping") else FALSE
    )
  )
}

as_fit <- function(fit, columns) {
  names(fit$coefficients) <- columns
  structure(fit, class = "shrinkage_fit")
}

as_path <- function(path, columns) {
  path$fits <- lapply(path$fits, as_fit, columns = columns)
  path$lambda <- vapply(path$fits, function(fit) fit$lambda, numeric(1))
  structure(path, class = "shrinkage_path")
}

#' Fit Gaussian lasso models
#'
#' @description Fit a Gaussian lasso model, a warm-start regularization path,
#' or a cross-validated path.
#'
#' @param x Finite base R numeric matrix or Matrix `dgCMatrix`.
#'   Rows represent observations and columns represent features.
#' @param y Finite numeric response vector with one value per row.
#' @param lambda Finite nonnegative penalty strengths in fitting order.
#'   `NULL` generates a decreasing log-spaced path. One value fits a single
#'   model in `lasso()`.
#' @param intercept Fit an unpenalized intercept.
#' @param standardize Scale by population standard deviations and center
#'   when fitting an intercept. The penalty uses the standardized scale.
#' @param tolerance Positive absolute KKT tolerance. The default uses a
#'   relative duality gap tolerance of 1e-6 at positive strengths and an absolute
#'   KKT tolerance of 1e-6 at zero strength.
#' @param max_iterations Maximum coordinate sweeps per fit.
#' @param nlambda Requested automatic path length.
#' @param lambda_min_ratio Automatic endpoint as a fraction of the initial
#'   strength. The default is 0.01 when there are fewer observations than
#'   features, and 0.0001 otherwise. In CV the smallest training fold determines
#'   the default.
#' @param early_stopping Allow automatic paths to stop on deviance.
#' @param foldid One-based consecutive fold IDs, with at least two folds.
#'   Supply either `foldid` or `nfolds`.
#' @param nfolds Number of generated folds, defaulting to the smaller of
#'   ten and the number of observations.
#' @param seed Nonnegative integer seed for generated folds.
#' @param rule Select the first minimum-error candidate or the first candidate
#'   within one standard error of that minimum. Put simpler candidates first.
#' @details
#'
#' The objective is the residual sum of squares divided by twice the number of
#' observations, plus lambda times the L1 norm of the optimization coefficients.
#' Cross-validation scores mean squared error on held-out rows and weights each
#' fold equally. Each fold fits its own training preprocessing and, for automatic
#' paths, its own starting lambda. Cross-validation requires convergence at every
#' candidate.
#'
#' @return
#'
#' A `shrinkage_fit` contains original-scale coefficients and intercept,
#' lambda, preprocessing, and diagnostics. A `shrinkage_path` contains
#' `fits`, their `lambda` values, path termination, and automatic path
#' diagnostics. A `shrinkage_cv` includes the full-data `path`, fold IDs,
#' matrices of fold scores and lambdas, mean scores, standard errors, automatic
#' relative strengths, and one-based selection indices.
#'
#' `coef()` includes the intercept as its first entry. Supply `newdata`
#' to `predict()` with features in training order. These methods return a
#' vector for a single fit or the selected CV fit. For paths they return a matrix
#' with one column per fitted lambda.
#'
#' @examples
#'
#' x <- cbind(weight = mtcars$wt, power = mtcars$hp)
#' fit <- lasso(x, mtcars$mpg, lambda = 0.5)
#' coef(fit)
#' predict(fit, x[1:3, , drop = FALSE])
#' path <- lasso(x, mtcars$mpg, nlambda = 5)
#' cv <- cv_lasso(x, mtcars$mpg, lambda = c(1, 0.5), nfolds = 3)
#' cv$lambda[cv$selected_index]
#'
#' @export
lasso <- function(x, y, lambda = NULL, intercept = TRUE, standardize = TRUE,
                  tolerance = NULL, max_iterations = 10000L, nlambda = 100L,
                  lambda_min_ratio = NULL, early_stopping = TRUE) {
  inputs <- fit_inputs(x, y, lambda, intercept, standardize, tolerance,
                       max_iterations, nlambda, lambda_min_ratio, early_stopping)
  result <- do.call(shrinkage_fit_native, c(inputs, list(cv = NULL)))
  if (length(lambda) == 1L) as_fit(result, colnames(x)) else as_path(result, colnames(x))
}

#' @rdname lasso
#' @export
cv_lasso <- function(x, y, lambda = NULL, foldid = NULL, nfolds = NULL,
                     seed = 0L, rule = c("minimum", "one_se"), intercept = TRUE,
                     standardize = TRUE, tolerance = NULL, max_iterations = 10000L,
                     nlambda = 100L, lambda_min_ratio = NULL) {
  inputs <- fit_inputs(x, y, lambda, intercept, standardize, tolerance,
                       max_iterations, nlambda, lambda_min_ratio, FALSE)
  if (is.null(foldid)) {
    if (is.null(nfolds)) nfolds <- min(10L, nrow(x))
    nfolds <- count(nfolds, "nfolds", 2L)
    seed <- count(seed, "seed", 0L)
  } else {
    if (!is.null(nfolds)) stop("supply either foldid or nfolds", call. = FALSE)
    if (!numeric_values(foldid) || !is.null(dim(foldid)) ||
        length(foldid) != nrow(x) ||
        any(foldid < 1 | foldid > nrow(x) | foldid != floor(foldid))) {
      stop("foldid must contain one positive fold ID per observation", call. = FALSE)
    }
    foldid <- as.integer(foldid)
  }
  rule <- match.arg(rule)
  cv <- list(foldid = foldid, nfolds = nfolds, seed = seed, one_se = rule == "one_se")
  result <- do.call(shrinkage_fit_native, c(inputs, list(cv = cv)))
  result$path <- as_path(result$path, colnames(x))
  result$rule <- rule
  result$lambda <- result$path$lambda
  result$fold_scores <- do.call(rbind, result$fold_scores)
  result$fold_lambdas <- do.call(rbind, result$fold_lambdas)
  structure(result, class = "shrinkage_cv")
}

#' @export
coef.shrinkage_fit <- function(object, ...) {
  coefficients <- object$coefficients
  if (is.null(names(coefficients))) {
    names(coefficients) <- paste0("V", seq_along(coefficients), recycle0 = TRUE)
  }
  c("(Intercept)" = object$intercept, coefficients)
}

#' @export
coef.shrinkage_path <- function(object, ...) {
  do.call(cbind, lapply(object$fits, coef))
}

selected_fit <- function(object) object$path$fits[[object$selected_index]]

#' @export
coef.shrinkage_cv <- function(object, ...) coef(selected_fit(object), ...)

predict_columns <- function(newdata, coefficients) {
  design(newdata)
  if (ncol(newdata) != nrow(coefficients) - 1L) {
    stop("newdata must have the same number of columns as the training matrix",
         call. = FALSE)
  }
  result <- as.matrix(newdata %*% coefficients[-1L, , drop = FALSE])
  result <- sweep(result, 2L, coefficients[1L, ], "+")
  if (!all(is.finite(result))) stop("nonfinite prediction", call. = FALSE)
  result
}

#' @export
predict.shrinkage_fit <- function(object, newdata, ...) {
  result <- predict_columns(newdata, matrix(coef(object), ncol = 1L))
  result[, 1L]
}

#' @export
predict.shrinkage_path <- function(object, newdata, ...) {
  predict_columns(newdata, coef(object))
}

#' @export
predict.shrinkage_cv <- function(object, newdata, ...) {
  predict(selected_fit(object), newdata, ...)
}
