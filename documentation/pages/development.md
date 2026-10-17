# Development

Use `devenv shell` to enter the environment pinned to Rust 1.89.0. The core Rust
library and Python extension support Rust 1.85. Some optional backends and
development dependencies require newer Rust, so use the pinned toolchain for
tests and benchmarks. CI checks the MSRV separately by building the compatible
library features and Python extension without development dependencies.
Documentation uses a separately pinned Diplodocus build and its
`nightly-2026-09-28` extraction toolchain. The pinned Diplodocus build includes
a small compatibility patch that excludes generated Python environments, build
checkouts, and Nix hook symlinks from its Rust input inventory. Its regression
test runs in documentation CI.

## Python package

The optional package lives in `packages/python`. Its native crate and Cargo
lockfile are independent of the root Rust package. Maintained Python source and
stubs under its `src/shrinkage` directory supply the Python reference. The
install task rebuilds the editable native extension so checks include Rust
source changes.

```console
task python:install
task python:check
task python:artifacts
```

The artifact check installs a wheel into a fresh environment, tests it outside
the checkout, and repeats the checks after rebuilding a source distribution. The
private native smoke function exercises Rust penalty evaluation, and the dense
input tests verify validation, layouts, snapshots, and ownership against each
installed artifact. Public lasso tests compare fitting, preprocessing,
diagnostics, and prediction with Rust fixtures and independent optimality
calculations. A concurrency test verifies GIL release and snapshot independence.
`task example` checks that the committed fixtures match the current Rust API.

Use `task format` to format Rust, Python, and Markdown. `task check` runs the
complete Rust and Python checks. The Python package does not install or build as
part of a plain root `cargo build`.

## Documentation

```console
task docs:serve
task docs:build
task docs:check
```

These tasks prepare the editable Python package and fetch the root Rust
dependencies before invoking Diplodocus. Native Rust extraction and doctests
then run with locked, offline dependency resolution. Documentation builds also
execute the Python package's doctest against the prepared checkout.

`task docs:check` validates the generated Worker package without deploying it.
To test deployment routing locally, run `npm --prefix documentation run preview`
after building. Generated site output, environments, and caches are ignored.

The documentation workflow builds pull requests and `main`, and deploys
successful `main` builds to `https://shrinkage.cc/`. It uses the repository's
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` secrets. Manual workflow runs
deploy only when they target `main`.
