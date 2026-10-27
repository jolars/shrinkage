# Contributing

Use `devenv shell` to enter the environment pinned to Rust 1.89.0. The core Rust
library and Python extension support Rust 1.85. Some optional backends and
development dependencies require newer Rust, so use the pinned toolchain for
tests and benchmarks. CI checks the MSRV separately by building the compatible
library features and Python extension without development dependencies.
Documentation uses `docs/diplodocus.yaml`, a separately pinned Diplodocus 0.6.0
build, and its `nightly-2026-09-28` extraction toolchain. The pinned Diplodocus
build includes compatibility patches for generated input exclusions and native
GFM math. The math patch also lives in the sibling Diplodocus checkout for
upstream release. Devenv and CI apply both patches to the same pinned 0.6.0
source revision. CI keeps its Python environment outside the checkout.

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

Edit this guide in the root `CONTRIBUTING.md`. Documentation tasks and CI copy
it to `docs/pages/contribution.md` before building the site.

```console
task docs:serve
task docs:build
task docs:check
task docs:figures
```

These tasks prepare the editable Python package and fetch the root Rust
dependencies before invoking Diplodocus. Native Rust extraction and doctests
then run with locked, offline dependency resolution. Documentation builds also
execute the Python package's doctest against the prepared checkout.

The Reference section lives under `docs/pages/reference`. Write inline math with
`$...$` and display equations with `$$...$$`. Diplodocus renders native KaTeX
during the build and bundles its CSS and fonts, including MathML for accessible
equations. Check the built pages for `math-fallback` markers after adding new
commands.

`task docs:figures` regenerates the committed SVG penalty regions with pinned
Matplotlib and NumPy dependencies. The generator fixes its metadata and SVG
identifiers and checks boundary, interior, and exterior samples.
`task docs:figures:check`, also run by `task docs:check`, verifies that the
committed figures are current. The page captions record the plotted strengths,
groups, and coefficient layouts.

`task docs:check` validates the generated Worker package without deploying it.
To test deployment routing locally, run `npm --prefix docs run preview` after
building. Generated site output, environments, and caches are ignored.

The documentation workflow builds every push and pull request, and deploys
successful `main` builds to `https://shrinkage.cc/`. It uses the repository's
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` secrets. Manual workflow runs
deploy only when they target `main`.
