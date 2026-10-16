"""Regularized statistical models backed by Rust.

This initial package establishes native installation and version reporting.
Fitting and prediction bindings are planned for subsequent milestones.

Examples
--------
>>> import shrinkage
>>> from importlib.metadata import version
>>> shrinkage.__version__ == version("shrinkage")
True
"""

from . import _shrinkage as _shrinkage

__version__ = "0.1.0"
__all__ = ["__version__"]
