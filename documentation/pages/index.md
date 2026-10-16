# Shrinkage

Shrinkage is a Rust library for regularized statistical models, with a Python
package under development. Both packages share numerical implementations and
statistical conventions, and each has its own version and release schedule.

The Rust library fits Gaussian lasso models with coordinate descent and supports
lasso, ridge, elastic net, and SLOPE through proximal gradient. Its typed and
runtime APIs also accept external smooth datafits and proximal penalties.

Start with [installation](installation.md), consult the [Rust API
reference](https://shrinkage.cc/packages/rust/), or read the [statistical
conventions](https://github.com/jolars/shrinkage/blob/main/DESIGN.md). The
[Python package](https://shrinkage.cc/packages/python/) currently provides
native installation, version reporting, and a private validated dense NumPy
input boundary. Python fitting and prediction are upcoming milestones.

For contributors, see [development](development.md) and [release
conventions](releases.md).
