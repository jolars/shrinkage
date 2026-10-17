"""Regularized statistical models backed by Rust.

The package establishes native installation, version reporting, and validated
dense NumPy input ownership. Fitting and prediction bindings are planned for
the next milestone.

Examples
--------
>>> import shrinkage
>>> from importlib.metadata import version
>>> shrinkage.__version__ == version("shrinkage")
True
"""

from . import _shrinkage as _shrinkage

__version__ = "0.2.0"
__all__ = ["__version__"]
