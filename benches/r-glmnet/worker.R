entry <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
.libPaths(c(file.path(dirname(entry), "library"), .libPaths()))
loadNamespace("Matrix")
role <- commandArgs(TRUE)[[1L]]

materialize <- function(context) {
  settings <- context$parameters
  n <- settings$n_samples
  p <- settings$n_features
  set.seed(settings$data_seed, kind = "Mersenne-Twister", normal.kind = "Inversion")
  x <- if (settings$density == 1) {
    matrix(rnorm(n * p), n, p)
  } else {
    Matrix::rsparsematrix(n, p, settings$density, rand.x = rnorm)
  }
  signals <- seq_len(min(20, p))
  y <- as.numeric(x[, signals, drop = FALSE] %*% rep_len(c(1, -1), length(signals))) + rnorm(n)
  y <- y - mean(y)
  lambda <- settings$lambda_fraction * max(abs(Matrix::crossprod(x, y))) / n
  if (!is.finite(lambda) || lambda <= 0) stop("The generated design must have positive lambda_max.")
  saveRDS(list(x = x, y = y, lambda = lambda), file.path(context$output_dir, "data.rds"),
          compress = FALSE)
  metewand::dataset(files = list(metewand::artifact("data.rds", "application/octet-stream")))
}

prepare <- function(context) {
  dir <- context$dataset_dir
  path <- file.path(dir, "data.rds")
  if (file.exists(path)) return(readRDS(path))
  data <- metewand::read_json(file.path(dir, "data.json"))
  list(x = do.call(rbind, lapply(data$x, unlist)), y = unlist(data$y), lambda = data$lambda)
}

evaluate <- function(context) {
  data <- prepare(context)
  beta <- unlist(metewand::read_json(file.path(context$result_dir, "result.json"))$data$beta)
  if (!is.numeric(beta) || length(beta) != ncol(data$x) || any(!is.finite(beta))) {
    return(metewand::evaluation(stats::setNames(list(), character()), "invalid"))
  }
  n <- length(data$y)
  r <- data$y - as.numeric(data$x %*% beta)
  gradient <- as.numeric(Matrix::crossprod(data$x, r)) / n
  lambda <- data$lambda
  scale <- lambda / max(lambda, abs(gradient))
  # A feasible residual-scaled dual gives this gap without subtracting objectives.
  gap <- sum(((1 - scale) * r)^2) / (2 * n) +
    sum(abs(beta) * pmax(0, lambda - sign(beta) * scale * gradient))
  metrics <- list(
    objective = sum(r^2) / (2 * n) + lambda * sum(abs(beta)),
    duality_gap = gap,
    relative_gap = gap / (sum(data$y^2) / (2 * n)),
    kkt_violation = max(ifelse(beta == 0, pmax(0, abs(gradient) - lambda),
                               abs(gradient - lambda * sign(beta)))),
    lambda = lambda
  )
  if (any(!is.finite(unlist(metrics)))) return(metewand::evaluation(stats::setNames(list(), character()), "invalid"))
  target <- context$semantics$one_shot_completion$relative_gap
  metewand::evaluation(metrics, if (metrics$relative_gap <= target) "reached" else "not_reached")
}

if (role == "dataset") {
  metewand::serve_dataset_materializer(materialize)
} else if (role == "evaluate") {
  metewand::serve_evaluator(evaluate)
} else {
  stopifnot(role %in% c("shrinkage", "glmnet"))
  loadNamespace(role)
  native <- getLoadedDLLs()[[role]][["path"]]
  software <- list(
    version = as.character(packageVersion(role)),
    native_sha256 = digest::digest(file = native, algo = "sha256"),
    r = R.version.string,
    matrix = as.character(packageVersion("Matrix")),
    blas = unname(extSoftVersion()[["BLAS"]])
  )
  if (role == "shrinkage") software$build <- metewand::read_json(file.path(dirname(entry), "build.json"))
  execute <- if (role == "shrinkage") {
    function(data, context) {
      shrinkage::shrinkage(
        data$x, data$y, penalty = shrinkage::penalty_l1(data$lambda),
        solver = "coordinate_descent", intercept = FALSE, normalization = "none",
        tolerance = 1e-11, max_iterations = context$observation_control$sweeps %||% 10000L
      )
    }
  } else {
    function(data, context) {
      fit <- glmnet::glmnet(
        data$x, data$y, family = "gaussian", alpha = 1, lambda = data$lambda,
        intercept = FALSE, standardize = FALSE, type.gaussian = "naive",
        control = list(thresh = context$observation_control$threshold %||% 1e-18)
      )
      list(fit = fit, lambda = data$lambda)
    }
  }
  `%||%` <- function(value, default) if (is.null(value)) default else value
  metewand::serve_implementation(
    prepare = prepare, execute = execute,
    result = function(fit, context) {
      if (role == "glmnet") {
        requested <- fit$lambda
        fit <- fit$fit
        if (length(fit$lambda) != 1L || !is.finite(fit$lambda) || fit$jerr != 0L ||
            !isTRUE(all.equal(fit$lambda, requested, tolerance = 1e-14))) {
          stop("glmnet did not return a solution at the requested lambda.")
        }
      }
      list(beta = as.list(as.numeric(stats::coef(fit))[-1L]))
    },
    statistics = function(fit, context) software,
    reset = function(state, context) NULL
  )
}
