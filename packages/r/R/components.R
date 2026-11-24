#' Specify a model loss
#'
#' @param link Link relating the linear predictor to the response mean.
#' @return A loss specification for [shrinkage()].
#' @export
loss_gaussian <- function() {
  structure(
    list(name = "gaussian", link = "identity"),
    class = "shrinkage_loss"
  )
}

#' @rdname loss_gaussian
#' @export
loss_binomial <- function(link = c("logit", "probit")) {
  structure(
    list(name = "binomial", link = match.arg(link)),
    class = "shrinkage_loss"
  )
}

penalty_spec <- function(name, ...) {
  structure(list(name = name, ...), class = "shrinkage_penalty")
}

strengths <- function(x, name, scalar = FALSE) {
  if (
    !numeric_values(x) ||
      !is.null(dim(x)) ||
      !length(x) ||
      any(x < 0) ||
      (scalar && length(x) != 1L)
  ) {
    stop(
      name,
      if (scalar) {
        " must be one finite nonnegative strength"
      } else {
        " must contain finite nonnegative strengths"
      },
      call. = FALSE
    )
  }
  as.double(x)
}

#' Specify a model penalty
#'
#' @param lambda Finite nonnegative penalty strength.
#' @param l1,l2 Strengths of the L1 norm and half the squared L2 norm.
#' @param fusion Strength of the absolute differences between coefficients.
#' @param weights Optional edge weights for the chain, nonincreasing rank
#'   weights for SLOPE, or one weight per group for group lasso.
#' @return A penalty specification for [shrinkage()].
#' @export
penalty_l1 <- function(lambda) {
  penalty_spec("l1", lambda = strengths(lambda, "lambda", TRUE))
}

#' @rdname penalty_l1
#' @export
penalty_ridge <- function(lambda) {
  penalty_spec("ridge", lambda = strengths(lambda, "lambda", TRUE))
}

#' @rdname penalty_l1
#' @export
penalty_elastic_net <- function(l1, l2) {
  penalty_spec(
    "elastic_net",
    l1 = strengths(l1, "l1", TRUE),
    l2 = strengths(l2, "l2", TRUE)
  )
}

#' @rdname penalty_l1
#' @export
penalty_clustered_lasso <- function(l1, fusion) {
  penalty_spec(
    "clustered_lasso",
    l1 = strengths(l1, "l1", TRUE),
    fusion = strengths(fusion, "fusion", TRUE)
  )
}

#' @rdname penalty_l1
#' @export
penalty_chain_fused_lasso <- function(l1, fusion, weights = NULL) {
  penalty_spec(
    "chain_fused_lasso",
    l1 = strengths(l1, "l1", TRUE),
    fusion = strengths(fusion, "fusion", TRUE),
    weights = if (!is.null(weights)) strengths(weights, "weights")
  )
}

#' @rdname penalty_l1
#' @export
penalty_slope <- function(weights) {
  weights <- strengths(weights, "weights")
  if (is.unsorted(-weights)) {
    stop("SLOPE weights must be nonincreasing", call. = FALSE)
  }
  penalty_spec("slope", weights = weights)
}

#' Specify a regularization path
#'
#' @param penalty A penalty specification. The automatic path varies a scalar
#'   strength while preserving the penalty shape, such as an elastic net's
#'   mixing ratio or a SLOPE rank profile.
#' @param nlambda Requested number of points.
#' @param lambda_min_ratio Fraction of the starting strength at the last point.
#'   The default is 0.01 when there are fewer observations than features, and
#'   0.0001 otherwise. In CV, the smallest training fold determines the default.
#' @param lambda_start Optional starting strength. By default it is computed
#'   from the data and penalty.
#' @param early_stopping Whether an automatic path can stop based on deviance.
#' @param penalties A nonempty list of complete penalty specifications in
#'   fitting order. Put simpler candidates first for the one-standard-error rule.
#' @return A path specification for the `penalty` argument of [shrinkage()].
#' @export
path_auto <- function(
  penalty = penalty_l1(1),
  nlambda = 100L,
  lambda_min_ratio = NULL,
  lambda_start = NULL,
  early_stopping = TRUE
) {
  check_penalty(penalty)
  if (!is.null(lambda_min_ratio)) {
    lambda_min_ratio <- strengths(lambda_min_ratio, "lambda_min_ratio", TRUE)
    if (lambda_min_ratio <= 0 || lambda_min_ratio > 1) {
      stop("lambda_min_ratio must be in (0, 1]", call. = FALSE)
    }
  }
  structure(
    list(
      name = "auto",
      penalties = list(penalty),
      nlambda = count(nlambda, "nlambda"),
      lambda_min_ratio = lambda_min_ratio,
      lambda_start = if (!is.null(lambda_start)) {
        strengths(lambda_start, "lambda_start", TRUE)
      },
      early_stopping = flag(early_stopping, "early_stopping")
    ),
    class = "shrinkage_path_spec"
  )
}

#' @rdname path_auto
#' @export
path_explicit <- function(penalties) {
  if (!is.list(penalties) || !length(penalties)) {
    stop(
      "penalties must be a nonempty list of penalty specifications",
      call. = FALSE
    )
  }
  invisible(lapply(penalties, check_penalty))
  structure(
    list(name = "explicit", penalties = unname(penalties)),
    class = "shrinkage_path_spec"
  )
}

check_penalty <- function(x) {
  if (!inherits(x, "shrinkage_penalty")) {
    stop("penalty must be a penalty specification", call. = FALSE)
  }
}

#' Specify cross-validation
#'
#' @param foldid One-based consecutive fold IDs, with at least two folds.
#'   Supply either `foldid` or `nfolds`.
#' @param nfolds Number of generated folds. The default is the smaller of ten
#'   and the number of observations.
#' @param seed Nonnegative integer seed for generated folds.
#' @param rule Select the first minimum-error candidate or the first candidate
#'   within one standard error of that minimum.
#' @param metric Held-out score. By default, Gaussian uses mean squared error
#'   and binomial uses log loss.
#' @param stratified Generate stratified folds for binary responses.
#' @return A cross-validation specification for [shrinkage()].
#' @export
cv_options <- function(
  foldid = NULL,
  nfolds = NULL,
  seed = 0L,
  rule = c("minimum", "one_se"),
  metric = c("auto", "mse", "mae", "log_loss", "deviance"),
  stratified = FALSE
) {
  stratified <- flag(stratified, "stratified")
  if (!is.null(foldid)) {
    if (!is.null(nfolds)) {
      stop("supply either foldid or nfolds", call. = FALSE)
    }
    if (stratified) {
      stop("stratified requires generated folds", call. = FALSE)
    }
    if (
      !numeric_values(foldid) ||
        !is.null(dim(foldid)) ||
        !length(foldid) ||
        any(foldid < 1 | foldid > length(foldid) | foldid != floor(foldid))
    ) {
      stop("foldid must contain positive consecutive fold IDs", call. = FALSE)
    }
    foldid <- as.integer(foldid)
  }
  structure(
    list(
      foldid = foldid,
      nfolds = if (!is.null(nfolds)) count(nfolds, "nfolds", 2L),
      seed = if (is.null(foldid)) count(seed, "seed", 0L),
      rule = match.arg(rule),
      metric = match.arg(metric),
      stratified = stratified
    ),
    class = "shrinkage_cv_options"
  )
}

#' @rdname loss_gaussian
#' @export
loss_multivariate_gaussian <- function() {
  structure(
    list(name = "multivariate_gaussian", link = "identity"),
    class = "shrinkage_loss"
  )
}

#' @rdname penalty_l1
#' @param groups A list of disjoint groups of one-based coefficient indices.
#' @param uncovered Whether coefficients outside the groups cause an error or
#'   are unpenalized.
#' @export
penalty_group_lasso <- function(
  lambda,
  groups,
  weights = rep(1, length(groups)),
  uncovered = c("reject", "unpenalized")
) {
  if (!is.list(groups)) {
    stop("groups must be a list of coefficient indices", call. = FALSE)
  }
  groups <- lapply(groups, function(group) {
    if (
      !numeric_values(group) ||
        !is.null(dim(group)) ||
        !length(group) ||
        any(group < 1 | group > .Machine$integer.max | group != floor(group))
    ) {
      stop("groups must contain positive integer indices", call. = FALSE)
    }
    as.integer(group)
  })
  if (
    !numeric_values(weights) ||
      !is.null(dim(weights)) ||
      any(weights < 0) ||
      length(weights) != length(groups)
  ) {
    stop(
      "weights must contain one finite nonnegative value per group",
      call. = FALSE
    )
  }
  penalty_spec(
    "group_lasso",
    lambda = strengths(lambda, "lambda", TRUE),
    groups = unname(groups),
    weights = as.double(weights),
    uncovered = match.arg(uncovered)
  )
}

#' Compose a model penalty
#'
#' @param penalty A complete penalty specification from [penalty_l1()] or
#'   another penalty constructor.
#' @param scale Finite nonnegative multiplier for the penalty value.
#' @param unpenalized Logical vector with one entry per coefficient.
#'   `TRUE` excludes a coefficient from penalization. The inner penalty
#'   must be L1, ridge, or elastic net.
#' @param axis For multivariate fits, apply the penalty to each response's
#'   feature vector (`"features"`), each feature's response vector
#'   (`"responses"`), or the entire coefficient matrix in column order
#'   (`"all"`). For groups and masks, indices refer to the vector on
#'   the chosen axis.
#' @return A penalty specification for [shrinkage()].
#' @export
penalty_scaled <- function(penalty, scale) {
  check_penalty(penalty)
  penalty_spec(
    "scaled",
    penalty = penalty,
    scale = strengths(scale, "scale", TRUE)
  )
}

#' @rdname penalty_scaled
#' @export
penalty_masked <- function(penalty, unpenalized) {
  check_penalty(penalty)
  if (
    !is.logical(unpenalized) || !is.null(dim(unpenalized)) || anyNA(unpenalized)
  ) {
    stop("unpenalized must be a logical vector", call. = FALSE)
  }
  penalty_spec("masked", penalty = penalty, unpenalized = unname(unpenalized))
}

#' @rdname penalty_scaled
#' @export
penalty_axis <- function(penalty, axis = c("features", "responses", "all")) {
  check_penalty(penalty)
  penalty_spec("axis", axis = match.arg(axis), penalty = penalty)
}
