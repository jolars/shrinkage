args <- commandArgs(TRUE)
if (length(args) < 3L) stop("Usage: run.R WORKSPACE EXPERIMENT OUTPUT [METEWAND RUN OPTIONS]")
workspace <- normalizePath(args[[1L]], mustWork = TRUE)
manifest <- file.path(workspace, "metewand.yaml")
output <- args[[3L]]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
output <- normalizePath(output, mustWork = TRUE)
metewand <- Sys.getenv("METEWAND", "metewand")
run <- function(..., stdout = "") system2(metewand, shQuote(c(...)), stdout = stdout)
status <- run("--output", "json", "run", args[[2L]], "--manifest", manifest,
              "--output-dir", file.path(output, "run"), args[-(1:3)])
if (dir.exists(file.path(output, "run"))) {
  export <- tempfile("analysis-", tmpdir = output)
  exported <- run("--output", "json", "export", "--output-dir", file.path(output, "run"),
                  "--destination", export, "--format", "json",
                  stdout = file.path(output, paste0(basename(export), ".json")))
  if (exported == 0L) message("Results: ", file.path(export, "results.json"))
  if (status == 0L) status <- exported
}
quit(status = status)
