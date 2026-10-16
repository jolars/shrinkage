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
    pkgs.ruff
    (inputs.diplodocus.packages.${pkgs.stdenv.hostPlatform.system}.default.overrideAttrs (old: {
      patches = (old.patches or [ ]) ++ [ ./documentation/diplodocus-generated-inputs.patch ];
    }))
  ];

  languages.python = {
    enable = true;
    uv.enable = true;
  };

  env.DIPLODOCUS_RUST_TOOLCHAIN = rustDocumentationToolchain;

  # NumPy wheels need zlib even when Python has not loaded its zlib module yet.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.zlib ];

  # npm's Linux binary names a conventional loader path unavailable on NixOS.
  env.MINIFLARE_WORKERD_PATH = lib.mkIf pkgs.stdenv.isLinux (
    pkgs.writeShellScript "shrinkage-workerd" ''
      exec ${pkgs.stdenv.cc.bintools.dynamicLinker} \
        "${config.devenv.root}/documentation/node_modules/@cloudflare/workerd-linux-${
          if pkgs.stdenv.hostPlatform.isAarch64 then "arm64" else "64"
        }/bin/workerd" "$@"
    ''
  );

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
      entry = "panache format --check --force-exclude";
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
