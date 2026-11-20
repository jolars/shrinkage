{
  config,
  inputs,
  lib,
  pkgs,
  ...
}:
let
  rustOverlay = config.lib.getInput {
    name = "rust-overlay";
    url = "github:oxalica/rust-overlay";
    attribute = "DIPLODOCUS_RUST_TOOLCHAIN";
    follows = [ "nixpkgs" ];
  };
  rustDocumentationToolchain =
    (rustOverlay.lib.mkRustBin { } pkgs.buildPackages).nightly."2026-09-28".minimal;
in
{
  packages = [
    pkgs.go-task
    pkgs.panache
    pkgs.taplo
    pkgs.nodejs
    pkgs.wasm-bindgen-cli
    pkgs.ruff
    pkgs.cargo-flamegraph
    pkgs.samply
    pkgs.hyperfine
    pkgs.llvmPackages.bintools
    (inputs.diplodocus.packages.${pkgs.stdenv.hostPlatform.system}.default.overrideAttrs (old: {
      patches = (old.patches or [ ]) ++ [
        ./docs/diplodocus-generated-inputs.patch
      ];
    }))
  ]
  ++ lib.optionals pkgs.stdenv.isLinux [
    pkgs.perf
    pkgs.valgrind
    pkgs.util-linux
  ];

  languages.python = {
    enable = true;
    uv.enable = true;
  };

  env.DIPLODOCUS_RUST_TOOLCHAIN = rustDocumentationToolchain;
  env.SHRINKAGE_CHROMIUM_EXECUTABLE = lib.mkIf pkgs.stdenv.isLinux (lib.getExe pkgs.chromium);
  env.PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = lib.mkIf pkgs.stdenv.isLinux "1";
  env.PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = lib.mkIf pkgs.stdenv.isLinux "true";

  # NumPy wheels need these shared libraries outside Nix's patched executables.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  # npm's Linux binary names a conventional loader path unavailable on NixOS.
  env.MINIFLARE_WORKERD_PATH = lib.mkIf pkgs.stdenv.isLinux (
    pkgs.writeShellScript "shrinkage-workerd" ''
      exec ${pkgs.stdenv.cc.bintools.dynamicLinker} \
        "${config.devenv.root}/docs/node_modules/@cloudflare/workerd-linux-${
          if pkgs.stdenv.hostPlatform.isAarch64 then "arm64" else "64"
        }/bin/workerd" "$@"
    ''
  );

  languages.r = {
    enable = true;
    package = pkgs.rWrapper.override {
      packages = [
        pkgs.rPackages.Matrix
        pkgs.rPackages.roxygen2
        pkgs.rPackages.testthat
        pkgs.rPackages.knitr
        pkgs.rPackages.rmarkdown
        pkgs.rPackages.quarto
      ];
    };
  };

  languages.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
  };

  git-hooks.hooks = {
    cargo-clippy = {
      enable = true;
      name = "cargo clippy";
      entry = "cargo clippy --locked --all-targets --all-features -- -D warnings";
      files = "\\.rs$|Cargo\\.(toml|lock)$";
      language = "system";
      pass_filenames = false;
    };

    cargo-fmt = {
      enable = true;
      name = "cargo fmt";
      entry = "cargo fmt --all -- --check";
      files = "\\.rs$";
      language = "system";
      pass_filenames = false;
    };

    panache-format = {
      enable = true;
      entry = "panache format --force-exclude";
      files = "\\.md$";
      language = "system";
    };

    panache-lint = {
      enable = true;
      entry = "panache lint --force-exclude";
      files = "\\.md$";
      language = "system";
    };
  };
}
