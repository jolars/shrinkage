# Repository guidelines

## Scope and structure

Shrinkage is a Rust 2024 library for regularized statistical models. The current
bootstrap exposes LazyMatrix through `src/lib.rs`; model fitting follows
`DESIGN.md`. `examples/normalization.rs` exercises the matrix dependency with
dense and sparse CSC input.

Start with modules in this crate. Use `name.rs` and `name/child.rs`, never
`mod.rs`. Add solver tests and benchmarks with the numerical implementations.

## Architecture

- Reuse LazyMatrix's matrix capabilities and normalization algebra. Solver
  state, residual offsets, statistical validation, and stopping criteria belong
  here.
- Keep default features free of matrix backends. The pinned LazyMatrix release
  defines supported backend versions; the sibling checkout may contain
  unpublished APIs. Do not commit a dependency on a sibling directory.
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
cargo test --locked --all-features
cargo run --locked --example normalization --features faer
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
```

`task check` also runs `panache format --check .` and `panache lint .`. Use
`task format` to format source and Markdown, `task lint` for Clippy and Panache,
and `task example` for the matrix integration check. Preserve the declared MSRV
unless a deliberate compatibility change raises it.

## Generated files and releases

- Track `Cargo.lock` and `devenv.lock`; regenerate them with Cargo and devenv.
- Keep CI's Panache version aligned with the package in the locked devenv.
- Install CI toolchains with direct `rustup` commands.
- Do not edit `.pre-commit-config.yaml`; devenv generates it.
- Use Conventional Commits. Versionary manages `CHANGELOG.md`; do not edit it
  manually. Release automation requires the `RELEASE_TOKEN` GitHub secret.
