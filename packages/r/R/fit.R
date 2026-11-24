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
  if (
    !numeric_values(x) ||
      length(x) != 1L ||
      x < minimum ||
      x > .Machine$integer.max ||
      x != floor(x)
  ) {
    stop(name, " must be an integer >= ", minimum, call. = FALSE)
  }
  as.integer(x)
}

design <- function(x) {
  if (is(x, "dgCMatrix")) {
    validObject(x)
    values <- slot(x, "x")
    if (!all(is.finite(values))) {
      stop("x must be finite", call. = FALSE)
    }
    return(list(
      dim = dim(x),
      values = values,
      offsets = slot(x, "p"),
      indices = slot(x, "i")
    ))
  }
  if (!is.matrix(x) || !numeric_values(x)) {
    stop("x must be a finite numeric matrix or dgCMatrix", call. = FALSE)
  }
  list(dim = dim(x), values = as.double(x), offsets = NULL)
}

as_penalty <- function(penalty) {
  if (!is.null(penalty$penalty)) {
    penalty$penalty <- as_penalty(penalty$penalty)
  }
  structure(penalty, class = "shrinkage_penalty")
}

as_fit <- function(fit, columns, loss, responses) {
  if (loss$name == "multivariate_gaussian") {
    fit$coefficients <- matrix(
      fit$coefficients,
      ncol = length(fit$intercept),
      dimnames = list(columns, responses)
    )
    names(fit$intercept) <- responses
  } else {
    names(fit$coefficients) <- columns
  }
  fit$penalty <- as_penalty(fit$penalty)
  fit$loss <- loss
  structure(fit, class = "shrinkage_fit")
}

as_path <- function(path, columns, loss, responses) {
  path$fits <- lapply(
    path$fits,
    as_fit,
    columns = columns,
    loss = loss,
    responses = responses
  )
  path$penalties <- lapply(path$fits, function(fit) fit$penalty)
  structure(path, class = "shrinkage_path")
}

#' @export
coef.shrinkage_fit <- function(object, ...) {
  coefficients <- object$coefficients
  if (is.matrix(coefficients)) {
    if (is.null(rownames(coefficients))) {
      rownames(coefficients) <- feature_names(nrow(coefficients))
    }
    return(rbind("(Intercept)" = object$intercept, coefficients))
  }
  if (is.null(names(coefficients))) {
    names(coefficients) <- feature_names(length(coefficients))
  }
  c("(Intercept)" = object$intercept, coefficients)
}

feature_names <- function(n) paste0("V", seq_len(n), recycle0 = TRUE)

stack_responses <- function(points) {
  first <- points[[1L]]
  array(
    unlist(points, use.names = FALSE),
    dim = c(dim(first), length(points)),
    dimnames = c(dimnames(first) %||% list(NULL, NULL), list(NULL))
  )
}

#' @export
coef.shrinkage_path <- function(object, ...) {
  points <- lapply(object$fits, coef)
  if (is.matrix(points[[1L]])) {
    stack_responses(points)
  } else {
    do.call(cbind, points)
  }
}

selected_fit <- function(object) {
  object$path$fits[[object$selected_index]]
}

#' @export
coef.shrinkage_cv <- function(object, ...) {
  coef(selected_fit(object), ...)
}

predict_columns <- function(newdata, coefficients) {
  design(newdata)
  if (ncol(newdata) != nrow(coefficients) - 1L) {
    stop(
      "newdata must have the same number of columns as the training matrix",
      call. = FALSE
    )
  }
  result <- as.matrix(newdata %*% coefficients[-1L, , drop = FALSE])
  result <- sweep(result, 2L, coefficients[1L, ], "+")
  if (!all(is.finite(result))) {
    stop("nonfinite prediction", call. = FALSE)
  }
  result
}

response_scale <- function(predictor, loss, type) {
  if (type == "link" || is.null(loss) || loss$name != "binomial") {
    return(predictor)
  }
  if (loss$link == "probit") {
    stats::pnorm(predictor)
  } else {
    stats::plogis(predictor)
  }
}

#' @export
predict.shrinkage_fit <- function(
  object,
  newdata,
  ...,
  type = c("link", "response")
) {
  coefficients <- coef(object)
  if (is.matrix(coefficients)) {
    match.arg(type)
    return(predict_columns(newdata, coefficients))
  }
  result <- predict_columns(newdata, matrix(coefficients, ncol = 1L))
  response_scale(result[, 1L], object$loss, match.arg(type))
}

#' @export
predict.shrinkage_path <- function(
  object,
  newdata,
  ...,
  type = c("link", "response")
) {
  loss <- object$fits[[1L]]$loss
  type <- match.arg(type)
  if (identical(loss$name, "multivariate_gaussian")) {
    return(stack_responses(lapply(
      object$fits,
      predict,
      newdata = newdata,
      type = type
    )))
  }
  response_scale(predict_columns(newdata, coef(object)), loss, type)
}

#' @export
predict.shrinkage_cv <- function(object, newdata, ...) {
  predict(selected_fit(object), newdata, ...)
}
