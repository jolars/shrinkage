# Shrinkage

Shrinkage is a Rust framework for composable regularized statistical models. The
planned API supports both lean, statically dispatched solvers and runtime
composition for R and Python packages.

This repository currently contains the project infrastructure and a verified
LazyMatrix integration example. Model fitting is not implemented yet. See
[DESIGN.md](DESIGN.md) for the architecture, statistical conventions, and
implementation milestones, and [TODO.md](TODO.md) for the implementation
checklist.

## Matrix dependency

The crate pins LazyMatrix 0.2.0 from crates.io. No matrix backend is enabled by
default; optional `faer` and `nalgebra` features enable its dense and sparse CSC
adapters. Backend-specific code uses the versions supported by that release.

Run the dense/sparse normalization example:

```console
cargo run --locked --example normalization --features faer
```

The example checks that lazy products preserve predictions after transforming
coefficients and the intercept back to the original scale. It does not fit a
model.

## Development

Enter the development environment and run the complete local check:

```console
devenv shell
task check
```

The toolchain is pinned to Rust 1.89.0, which is also the minimum supported Rust
version. `Cargo.lock` and `devenv.lock` are tracked for reproducible
development. CI runs the same Rust checks, example, and Panache checks as
`task check`.

Useful focused commands are:

```console
task format
task lint
task test
```

Modules use `name.rs` and `name/child.rs`; the repository does not use `mod.rs`.

## Releases

Versionary prepares release pull requests after CI passes on `main` and manages
`CHANGELOG.md`. Before enabling releases on GitHub, configure the
`RELEASE_TOKEN` repository secret with access to contents, pull requests, and
issues. Registry publication is not configured.

## License

MIT. See [LICENSE](LICENSE).
