args <- commandArgs(TRUE)
if (!length(args) %in% 1:2) stop("Usage: prepare.R WORKSPACE [checkout|published]")
mode <- if (length(args) == 2L) args[[2L]] else "checkout"
stopifnot(mode %in% c("checkout", "published"))
stopifnot(packageVersion("glmnet") >= "5.1", packageVersion("metewand") >= "0.4.0")
root <- normalizePath(".", winslash = "/")
workspace <- args[[1L]]
if (dir.exists(workspace)) stop("Choose a new benchmark workspace.")
dir.create(workspace, recursive = TRUE)
workspace <- normalizePath(workspace)
lib <- file.path(workspace, "library")
dir.create(lib)
run <- function(command, args) {
  if (system2(command, args) != 0L) stop(command, " failed.")
}
if (!all(file.copy(file.path("benches/r-glmnet", c("metewand.yaml", "worker.R")), workspace))) {
  stop("Could not stage the benchmark sources.")
}
staging <- tempfile("shrinkage-benchmark-")
run("Rscript", c("packages/r/tools/package.R", "stage", shQuote(staging)))
source <- file.path(staging, "shrinkage")
manifest <- file.path(source, "src/rust/Cargo.toml")
if (mode == "checkout") {
  lines <- readLines(manifest)
  dependency <- grepl("^shrinkage-core = ", lines)
  stopifnot(sum(dependency) == 1)
  lines[dependency] <- sprintf(
    'shrinkage-core = { package = "shrinkage", path = %s, features = ["faer_v0_24"] }',
    encodeString(root, quote = '"')
  )
  writeLines(lines, manifest)
  run("cargo", c("update", "--manifest-path", shQuote(manifest), "-p", "shrinkage"))
}
run("R", c("CMD", "INSTALL", "--no-docs", "--no-html", "-l", shQuote(lib), shQuote(source)))
packages <- c("metewand", "glmnet", "Matrix", "digest")
db <- installed.packages()
deps <- unique(c(packages, unlist(tools::package_dependencies(packages, db, recursive = TRUE))))
for (pkg in deps) {
  if (pkg == "R" || pkg %in% rownames(installed.packages(priority = "base"))) next
  path <- find.package(pkg)
  if (!file.copy(path, lib, recursive = TRUE)) stop("Cannot stage ", pkg)
}
git <- function(args) system2("git", args, stdout = TRUE)
jsonlite::write_json(list(
  core = mode,
  revision = git(c("rev-parse", "HEAD")),
  dirty = length(git(c("status", "--porcelain"))) > 0L,
  lock_sha256 = digest::digest(file = file.path(dirname(manifest), "Cargo.lock"), algo = "sha256"),
  rust = system2("rustc", "--version", stdout = TRUE)
), file.path(workspace, "build.json"), auto_unbox = TRUE)
unlink(staging, recursive = TRUE)
