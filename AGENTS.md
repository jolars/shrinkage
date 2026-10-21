# Repository guidelines

## Scope and structure

Shrinkage is a Rust 2024 library for regularized statistical models. `Lasso`
provides the convenience API in `src/lasso.rs`; `src/solver/coordinate.rs` owns
Gaussian coordinate descent and its residual state. Typed composition, datafits,
penalties, and fitted results live in their respective modules. Model fitting
follows `DESIGN.md`. The `normalization` and `lasso` examples exercise dense and
sparse CSC input through LazyMatrix 0.5.0. Optional `faer_v0_24`,
`nalgebra_v0_34`, `ndarray_v0_17`, and `sprs_v0_11` features expose its matrix
integrations. The short names `faer`, `nalgebra`, `ndarray`, and `sprs` are
aliases. sprs input uses LazyMatrix's checked `SprsCsc` wrapper.

`experimental-block-reader` enables a provisional buffered proximal consumer in
`src/experimental.rs`. `zarrs_v0_22` and its `zarrs` alias support the
file-backed demonstration through the published LazyMatrix `ReadBlock`
capability. Solver iteration stays in `src/solver/proximal.rs`.

`Irls` in `src/solver/irls.rs` fits Gaussian and binomial models through
`DiagonalHessianDatafit` and `CoordinatePenalty`. It uses weighted coordinate
sweeps, reusable working sets, and full original-loss KKT checks, with results
in `IrlsFit`. Paths reuse its workspace. IRLS is exposed through the typed Rust
API; the Gaussian `Lasso` convenience API retains its specialized solver.

Start with modules in this crate. Use `name.rs` and `name/child.rs`, never
`mod.rs`. Add solver tests and benchmarks with the numerical implementations.

The optional Python package lives in `packages/python` as a separate Cargo
workspace with maintained Python source and native stubs. Keep it out of the
root Rust dependency graph. `documentation` contains the shared Diplodocus site
and Cloudflare Worker configuration.

## Architecture

- Reuse LazyMatrix's matrix capabilities and normalization algebra. Solver
  state, residual offsets, statistical validation, and stopping criteria belong
  here.
- Keep default features free of matrix backends. The pinned LazyMatrix release
  defines supported backend versions; the sibling checkout may contain
  unpublished APIs. Do not commit a dependency on a sibling directory.
- Gate tests, doctests, examples, and benchmarks on versioned backend features.
  Test the versioned features and their short aliases independently.
- Preserve the penalty scale, loss normalization, and intercept conventions in
  `DESIGN.md`. Back-transforming coefficients must preserve predictions.
- Keep runtime dispatch outside scalar loops. Establish fallible operations and
  reusable workspaces before stabilizing solver interfaces.
- Add a failing regression test before fixing observable behavior.

## Validation

Use `devenv shell` for the pinned Rust 1.89.0 toolchain. Run `task check` for
the complete check. Its Rust commands are:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo build --locked --no-default-features
cargo test --locked --no-default-features
cargo test --locked --no-default-features --features faer_v0_24
cargo test --locked --no-default-features --features nalgebra_v0_34
cargo test --locked --no-default-features --features ndarray_v0_17
cargo test --locked --no-default-features --features sprs_v0_11
cargo test --locked --no-default-features --features faer
cargo test --locked --no-default-features --features nalgebra
cargo test --locked --no-default-features --features ndarray
cargo test --locked --no-default-features --features sprs
cargo test --locked --no-default-features --features zarrs_v0_22
cargo test --locked --no-default-features --features zarrs
cargo test --locked --no-default-features --features experimental-block-reader,zarrs_v0_22
cargo test --locked --all-features
cargo run --locked --example normalization --features faer_v0_24
cargo run --locked --example lasso --features faer_v0_24
cargo run --locked --example binomial --features faer_v0_24
cargo run --locked --example multivariate --features faer_v0_24
cargo run --locked --example zarrs_buffered --features experimental-block-reader,zarrs_v0_22
cargo bench --locked --no-default-features --features faer_v0_24 --bench lasso_heap
cargo bench --locked --no-default-features --features faer_v0_24 --bench irls_heap
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
```

`task check` also runs `panache format --check .` and `panache lint .`. Use
`task format` to format source and Markdown, `task lint` for Clippy and Panache,
and `task example` for the matrix integration check. Preserve the declared MSRV
unless a deliberate compatibility change raises it.

`task check` also checks the Python package, installed wheels, a source
distribution rebuilt outside the checkout, and independent Versionary release
scenarios. Use `task python:check` for binding checks and `task docs:check` for
the shared site and Worker package. Documentation uses a separate pinned Rust
extraction toolchain; do not raise the library's MSRV to match it.

## Generated files and releases

- Track `Cargo.lock` and `devenv.lock`; regenerate them with Cargo and devenv.
- Keep CI's Panache version aligned with the package in the locked devenv.
- Install CI toolchains with direct `rustup` commands.
- Do not edit `.pre-commit-config.yaml`; devenv generates it.
- Use Conventional Commits. Versionary manages `CHANGELOG.md`; do not edit it
  manually. Release automation requires the `RELEASE_TOKEN` GitHub secret.
- Rust and Python versions advance independently. Keep Python's version, native
  manifest, and lockfile synchronized through its package-scoped Versionary
  rules. Versionary 1.6 composite strategies do not scope secondary manifests
  correctly for nested packages.
