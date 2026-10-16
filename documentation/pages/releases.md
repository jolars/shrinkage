# Release conventions

Versionary manages Rust and Python versions, changelogs, release PRs, and GitHub
release metadata. Package publication to crates.io and PyPI will be added in the
distribution milestone.

  | Package | Version sources                                                                            | Changelog                      | Tag                           |
  | ------- | ------------------------------------------------------------------------------------------ | ------------------------------ | ----------------------------- |
  | Rust    | Root `Cargo.toml`                                                                          | Root `CHANGELOG.md`            | `v<version>`                  |
  | Python  | `packages/python/pyproject.toml`, its native `Cargo.toml`, and `src/shrinkage/__init__.py` | `packages/python/CHANGELOG.md` | `shrinkage-python-v<version>` |

The Python strategy updates its project metadata and Python lockfile.
Package-scoped artifact rules update its public version constant, native
manifest, and Cargo lockfile. Versionary 1.6 resolves secondary composite
strategies against the repository root, so explicit rules keep these updates
inside the Python package. These versions identify the Python package, whose
version can differ from the bundled Rust core's version.

Each package normally receives its own release PR. Python-only and
documentation-only commits do not trigger a Rust release. A Rust release does
not automatically bump Python. A Python release builds the core source at its
own release commit, including core changes since the preceding Python release.
When intentionally shipping core changes through Python, refresh its Cargo
lockfile and include a release-worthy commit in `packages/python` so Versionary
can detect that release.

The Rust strategy refreshes Cargo lockfiles throughout the repository. Updating
the root crate version can therefore update Python's lockfile. If simultaneously
pending releases write the same files, Versionary groups them into one release
PR while keeping their package versions distinct. Separate lockfiles preserve
the lean root build but do not prevent every release dependency.

CI installs the pinned Versionary release and runs it only after Rust and Python
checks pass on `main`. The existing `RELEASE_TOKEN` secret authorizes its
release workflow. Do not edit generated changelogs or Versionary state by hand.

The documentation site follows `main` and shows the version of each extracted
package. Its `dev` footer describes the site build rather than a shared package
version. Historical documentation archives are deferred.
