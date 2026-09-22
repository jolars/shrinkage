{
  pkgs,
  ...
}:

{
  packages = [
    pkgs.go-task
    pkgs.panache
    pkgs.taplo
  ];

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
