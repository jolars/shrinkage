"""Regularized statistical models backed by Rust.

Fit dense Gaussian lasso with owned NumPy inputs, original-scale predictions,
fitted preprocessing, and explicit convergence diagnostics.

Examples
--------
>>> import shrinkage
>>> from importlib.metadata import version
>>> shrinkage.__version__ == version("shrinkage")
True
>>> import numpy as np
>>> x = np.array([[0.0], [1.0], [2.0]])
>>> fit = shrinkage.Lasso(0.1).fit(x, np.array([1.0, 3.0, 5.0]))
>>> fit.termination
'converged'
>>> fit.predict(x).shape
(3,)
"""

from . import _shrinkage as _shrinkage
from .lasso import (
    Lasso as Lasso,
)
from .lasso import (
    LassoFit as LassoFit,
)
from .lasso import (
    Normalization as Normalization,
)
from .lasso import (
    Preprocessing as Preprocessing,
)
from .lasso import (
    StoppingCriterion as StoppingCriterion,
)
from .lasso import (
    Termination as Termination,
)

__version__ = "0.2.0"
__all__ = [
    "Lasso",
    "LassoFit",
    "Normalization",
    "Preprocessing",
    "StoppingCriterion",
    "Termination",
    "__version__",
]
