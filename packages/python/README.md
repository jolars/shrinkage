# Shrinkage for Python

Shrinkage provides regularized statistical models backed by Rust. This first
Python milestone provides an installable native extension and version reporting.
Fitting, prediction, and NumPy input support will follow in later milestones.

The distribution and import names are both `shrinkage`. Build from the
repository with Rust 1.89 or newer and standard CPython 3.11 or newer:

```console
uv sync --locked --project packages/python
uv run --locked --project packages/python python -c "import shrinkage; print(shrinkage.__version__)"
```

The compatibility target is CPython 3.11–3.15 on Linux x86_64, Windows x86_64,
and macOS arm64 and x86_64. CI builds and imports an interpreter-specific wheel
for each combination. Free-threaded Python and other interpreters are outside
the initial support matrix. Wheels are not published yet.

See [the documentation](https://shrinkage.cc/) for the Rust API, development
instructions, and independent Rust and Python release conventions.
