package <- function(mode) {
  root <- normalizePath(".", winslash = "/")
  output <- file.path(root, "dist")
  dir.create(output, showWarnings = FALSE)
  staging <- if (mode == "stage") {
    file.path(root, ".build", "r")
  } else {
    tempfile("shrinkage-r-")
  }
  if (dir.exists(file.path(staging, "shrinkage"))) {
    stop("The staging directory already exists.")
  }
  dir.create(staging, recursive = TRUE, showWarnings = FALSE)
  if (mode != "stage") {
    on.exit(unlink(staging, recursive = TRUE))
  }
  source <- file.path(staging, "shrinkage")
  dir.create(source)
  entries <- c(
    "DESCRIPTION",
    "LICENSE",
    "NAMESPACE",
    ".Rbuildignore",
    "README.md",
    "R",
    "man",
    "tests",
    "tools",
    "configure",
    "configure.win",
    "cleanup",
    "cleanup.win"
  )
  if (file.exists("packages/r/NEWS.md")) {
    entries <- c(entries, "NEWS.md")
  }
  copy <- function(files, destination) {
    if (!all(file.copy(files, destination, recursive = TRUE))) {
      stop("Could not stage the package.")
    }
  }
  copy(file.path("packages/r", entries), source)
  dir.create(file.path(source, "src", "rust"), recursive = TRUE)
  copy(
    file.path(
      "packages/r/src",
      c("entrypoint.c", "Makevars.in", "Makevars.win.in", "shrinkage-win.def")
    ),
    file.path(source, "src")
  )
  rust <- file.path(source, "src", "rust")
  copy(
    file.path(
      "packages/r/src/rust",
      c("Cargo.toml", "Cargo.lock", "src", "document.rs")
    ),
    rust
  )
  if (mode == "stage") {
    return(invisible(source))
  }

  r <- file.path(
    R.home("bin"),
    if (.Platform$OS.type == "windows") "R.exe" else "R"
  )
  run <- function(...) {
    status <- system2(r, c("CMD", ...))
    if (status != 0L) {
      for (log in c(
        "shrinkage.Rcheck/00install.out",
        "shrinkage.Rcheck/tests/testthat.Rout.fail"
      )) {
        if (file.exists(log)) cat(readLines(log), sep = "\n")
      }
      stop("R CMD exited with status ", status)
    }
  }
  if (mode == "install") {
    run("INSTALL", shQuote(source))
    return(invisible(NULL))
  }
  previous <- setwd(staging)
  on.exit(setwd(previous), add = TRUE, after = FALSE)
  run("build", "--no-build-vignettes", "shrinkage")
  version <- read.dcf(file.path(source, "DESCRIPTION"), fields = "Version")[[
    1L
  ]]
  archive <- paste0("shrinkage_", version, ".tar.gz")
  if (mode == "check") {
    # Keep the linked-symbol scan. Static archive members also contain the
    # Rust runtime, and scanning them flags its exit and abort entry points.
    Sys.setenv(`_R_CHECK_COMPILED_CODE_USE_OBJECTS_SYMBOL_TABLES_` = "false")
    run("check", "--no-manual", shQuote(archive))
    log <- readLines("shrinkage.Rcheck/00check.log")
    if (any(grepl("^Status:.*(WARNING|ERROR)", log))) {
      stop("R CMD check reported a warning or error.")
    }
  }
  if (!file.copy(archive, output, overwrite = TRUE)) {
    stop("Could not copy ", archive)
  }
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L || !args %in% c("build", "check", "install", "stage")) {
  stop("Usage: Rscript packages/r/tools/package.R {build|check|install|stage}")
}
package(args)
