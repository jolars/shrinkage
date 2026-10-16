# Development

Use `devenv shell` to enter the pinned environment. The Rust library and Python
extension build with Rust 1.89. Documentation uses a separately pinned
Diplodocus build and its `nightly-2026-09-28` extraction toolchain. The pinned
Diplodocus build includes a small compatibility patch that excludes generated
Python environments, build checkouts, and Nix hook symlinks from its Rust input
inventory. Its regression test runs in documentation CI.

## Python package

The optional package lives in `packages/python`. Its native crate and Cargo
lockfile are independent of the root Rust package. Maintained Python source and
stubs under its `src/shrinkage` directory supply the Python reference.

```console
task python:install
task python:check
task python:artifacts
```

The artifact check installs a wheel into a fresh environment, tests it outside
the checkout, and repeats the checks after rebuilding a source distribution. The
private native smoke function exercises Rust penalty evaluation without
establishing a public numerical Python interface.

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
