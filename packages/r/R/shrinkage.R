#' Fit a regularized model
#'
#' @param x Finite numeric matrix or Matrix `dgCMatrix`. Rows are observations
#'   and columns are features.
#' @param y Finite numeric response vector with one value per observation.
#'   Binomial responses must be in `[0, 1]`. For multivariate Gaussian loss,
#'   supply a numeric matrix with one response per column.
#' @param loss A loss specification. See [loss_gaussian()].
#' @param penalty A complete penalty or a path specification. See [penalty_l1()]
#'   and [path_auto()].
#' @param cv Optional cross-validation specification from [cv_options()].
#' @param solver A solver name. The default chooses a compatible solver for the
#'   loss, penalty, and fitting operation.
#' @param intercept Fit an unpenalized intercept.
#' @param normalization Training-column normalization. By default, scale by
#'   population standard deviations and center when fitting an intercept.
#' @param tolerance Positive convergence tolerance. The default and criterion
#'   depend on the solver.
#' @param max_iterations Positive iteration limit.
#' @return
#' A `shrinkage_fit` for one penalty, a `shrinkage_path` for a path
#' specification, or a `shrinkage_cv` when requesting cross-validation.
#' @details
#' Penalties apply to the coefficients on the normalized scale. Returned
#' coefficients and predictions use the original scale. `predict()` uses the
#' linear predictor by default. Use `type = "response"` for binomial
#' probabilities.
#'
#' Paths retain the complete penalties in fitting order. A `lambda` vector is
#' available for automatic paths and when every explicit point has a scalar L1
#' or ridge strength. Explicit paths can mix different penalty types with
#' proximal gradient. Group, scaled, and masked penalties require explicit
#' candidates for paths.
#'
#' The probit link uses IRLS with L1, ridge, or elastic net. Multivariate
#' Gaussian uses proximal gradient and applies the penalty by feature vector
#' within each response by default. Use [penalty_axis()] to choose another axis.
#' These two losses support single fits and paths. For multivariate fits,
#' `coef()` and `predict()` return matrices with one column per response. A path
#' adds a third dimension for its points.
#'
#' Cross-validation supports Gaussian and binomial logit losses.
#' Cross-validation returns the selected full-data fit through `coef()` and
#' `predict()`. It also retains the full-data path and held-out scores.
#' @examples
#' x <- cbind(weight = mtcars$wt, power = mtcars$hp)
#' fit <- shrinkage(x, mtcars$mpg, penalty = penalty_l1(0.5))
#' coef(fit)
#' logistic <- shrinkage(x, mtcars$am, loss_binomial(), penalty_ridge(0.1))
#' predict(logistic, x[1:3, , drop = FALSE], type = "response")
#' @export
shrinkage <- function(
  x,
  y,
  loss = loss_gaussian(),
  penalty = path_auto(),
  cv = NULL,
  solver = c(
    "auto",
    "coordinate_descent",
    "irls",
    "hybrid_coordinate_descent",
    "proximal_gradient"
  ),
  intercept = TRUE,
  normalization = c(
    "auto",
    "none",
    "standardize",
    "center",
    "min_max",
    "max_abs",
    "l1",
    "l2"
  ),
  tolerance = NULL,
  max_iterations = 10000L
) {
  matrix <- design(x)
  if (!inherits(loss, "shrinkage_loss")) {
    stop("loss must be a loss specification", call. = FALSE)
  }
  multivariate <- identical(loss$name, "multivariate_gaussian")
  if (multivariate) {
    if (
      !is.matrix(y) || !numeric_values(y) || nrow(y) != nrow(x) || !length(y)
    ) {
      stop(
        "y must be a finite response matrix with one row per observation",
        call. = FALSE
      )
    }
  } else if (
    !numeric_values(y) || !is.null(dim(y)) || length(y) != nrow(x) || !length(y)
  ) {
    stop(
      "y must be a finite vector with one value per observation",
      call. = FALSE
    )
  }
  if (!is.null(tolerance)) {
    tolerance <- strengths(tolerance, "tolerance", TRUE)
    if (tolerance == 0) stop("tolerance must be positive", call. = FALSE)
  }
  if (!inherits(penalty, "shrinkage_path_spec")) {
    check_penalty(penalty)
    penalty <- list(name = "single", penalties = list(penalty))
  }
  if (!is.null(cv)) {
    if (!inherits(cv, "shrinkage_cv_options")) {
      stop("cv must be a cross-validation specification", call. = FALSE)
    }
    if (!is.null(cv$foldid) && length(cv$foldid) != nrow(x)) {
      stop("foldid must contain one fold ID per observation", call. = FALSE)
    }
    if (is.null(cv$foldid) && is.null(cv$nfolds)) {
      cv$nfolds <- min(10L, nrow(x))
    }
    if (identical(penalty$name, "single")) penalty$name <- "explicit"
  }
  options <- list(
    solver = match.arg(solver),
    normalization = match.arg(normalization),
    intercept = flag(intercept, "intercept"),
    tolerance = tolerance,
    max_iterations = count(max_iterations, "max_iterations"),
    nresponses = if (multivariate) ncol(y) else 1L
  )
  result <- shrinkage_fit_native(
    matrix,
    as.double(y),
    loss,
    penalty,
    options,
    cv
  )

  if (!is.null(cv)) {
    result$path <- as_path(result$path, colnames(x), loss, colnames(y))
    result$lambda <- result$path$lambda
    result$fold_scores <- do.call(rbind, result$fold_scores)
    if (!is.null(result$fold_lambdas)) {
      result$fold_lambdas <- do.call(rbind, result$fold_lambdas)
    }
    return(structure(result, class = "shrinkage_cv"))
  }
  if (identical(penalty$name, "single")) {
    as_fit(result, colnames(x), loss, colnames(y))
  } else {
    as_path(result, colnames(x), loss, colnames(y))
  }
}
