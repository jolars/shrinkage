cargo <- Sys.getenv("CARGO", unset = Sys.which("cargo"))
rustc <- Sys.getenv("RUSTC", unset = Sys.which("rustc"))
for (tool in c(cargo, rustc)) {
  if (!nzchar(tool)) {
    stop("Shrinkage requires Cargo and rustc >= 1.89.")
  }
  version <- system2(tool, "--version", stdout = TRUE)
  if (
    utils::compareVersion(sub("^[^ ]+ ([0-9.]+).*", "\\1", version), "1.89") < 0
  ) {
    stop("Shrinkage requires Cargo and rustc >= 1.89, found ", version)
  }
}
windows <- .Platform$OS.type == "windows"
template <- if (windows) "src/Makevars.win.in" else "src/Makevars.in"
text <- readLines(template)
replacements <- list(
  "@CARGO@" = shQuote(cargo, type = "sh"),
  "@RUSTC@" = shQuote(rustc, type = "sh"),
  "@NATIVE_LIBS@" = if (grepl("darwin", R.version$os)) {
    ""
  } else {
    "-ldl -lpthread -lm"
  },
  "@TARGET_DIR@" = if (windows) {
    tempfile("sr-", tmpdir = normalizePath(dirname(tempdir()), winslash = "/"))
  } else {
    "rust/target"
  }
)
for (key in names(replacements)) {
  text <- gsub(key, replacements[[key]], text, fixed = TRUE)
}
writeLines(text, sub("\\.in$", "", template))
