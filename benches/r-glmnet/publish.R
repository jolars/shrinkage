args <- commandArgs(TRUE)
if (length(args) != 1L) stop("Usage: publish.R RESULTS_JSON")
snapshot <- jsonlite::read_json(args[[1L]])
target <- 1e-8
methods <- c("shrinkage", "glmnet")
candidates <- snapshot$candidates
field <- function(rows, name) vapply(rows, `[[`, character(1), name)
stopifnot(
  isTRUE(snapshot$completeness$complete),
  length(candidates) == 2L,
  all(field(candidates, "experiment") == "vignette"),
  setequal(field(candidates, "implementation"), methods)
)
observations <- Filter(function(row) isTRUE(row$selected) && !isTRUE(row$warmup),
                       snapshot$observations)
hashes <- vapply(observations, function(row) {
  snapshot$provenance[[row$provenance_id]]$dataset$output$sha256
}, character(1))
stopifnot(length(unique(hashes)) == 1L)
table <- do.call(rbind, lapply(candidates, function(candidate) {
  method <- candidate$implementation
  settings <- candidate$parameters$dataset
  stopifnot(settings$n_samples == 4096, settings$n_features == 1536,
            settings$density == 1, settings$data_seed == 17,
            settings$lambda_fraction == 0.1, candidate$measurement_repetitions == 5)
  rows <- Filter(function(row) identical(row$implementation, method), observations)
  points <- do.call(rbind, lapply(rows, function(row) {
    stopifnot(row$timing_scope == "execute_only",
              row$profile$delivery == "fresh_sequence",
              is.finite(row$implementation_time_ns), row$implementation_time_ns > 0)
    provenance <- snapshot$provenance[[row$provenance_id]]
    stopifnot(is.finite(row$metrics$relative_gap), row$metrics$relative_gap >= 0,
              is.finite(row$metrics$lambda), row$metrics$lambda > 0)
    control <- row$profile$observation_control
    data.frame(
      method = method, version = row$statistics$version,
      control = if (method == "shrinkage") control$sweeps else control$threshold,
      repetition = row$measurement_index, fit_ms = row$implementation_time_ns / 1e6,
      relative_gap = row$metrics$relative_gap, lambda = row$metrics$lambda,
      reached = row$metrics$relative_gap <= target,
      dataset_sha256 = hashes[[1L]], native_sha256 = row$statistics$native_sha256,
      r = row$statistics$r, metewand = provenance$tool_version,
      os = provenance$host$os, cpu = if (is.null(provenance$host$cpu_model)) {
        provenance$host$architecture
      } else {
        provenance$host$cpu_model
      },
      core = if (method == "shrinkage") row$statistics$build$core else "",
      revision = if (method == "shrinkage") row$statistics$build$revision else "",
      dirty = if (method == "shrinkage") row$statistics$build$dirty else NA
    )
  }))
  repetitions <- seq_len(candidate$measurement_repetitions) - 1L
  stopifnot(
    length(unique(points$native_sha256)) == 1L,
    all(vapply(split(points$repetition, points$control), function(x) {
      identical(sort(as.integer(x)), repetitions)
    }, logical(1))),
    setequal(points$repetition[points$reached], repetitions)
  )
  points
}))
stopifnot(length(unique(table$lambda)) == 1L)
destination <- "packages/r/vignettes/lasso-convergence.csv"
temporary <- tempfile(tmpdir = dirname(destination))
utils::write.csv(table, temporary, row.names = FALSE)
if (!file.rename(temporary, destination)) stop("Could not update the vignette measurements.")
