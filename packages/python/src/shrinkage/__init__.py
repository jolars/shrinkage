"""Regularized statistical models backed by Rust.

Compose Gaussian or binomial models with complete penalties and explicit
solvers. Results own original-scale parameters and convergence diagnostics.

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
from ._common import (
    Normalization as Normalization,
)
from ._common import (
    Preprocessing as Preprocessing,
)
from ._common import (
    StoppingCriterion as StoppingCriterion,
)
from ._common import (
    Termination as Termination,
)
from .datafit import Binomial as Binomial
from .datafit import Gaussian as Gaussian
from .fit import (
    HybridFit as HybridFit,
)
from .fit import (
    ProximalFit as ProximalFit,
)
from .fit import (
    ScreeningDiagnostics as ScreeningDiagnostics,
)
from .fit import (
    WorkingSetDiagnostics as WorkingSetDiagnostics,
)
from .lasso import Lasso as Lasso
from .lasso import LassoFit as LassoFit
from .penalty import (
    ChainFusedLasso as ChainFusedLasso,
)
from .penalty import (
    ClusteredLasso as ClusteredLasso,
)
from .penalty import (
    Slope as Slope,
)
from .problem import Problem as Problem
from .solver import (
    Hybrid as Hybrid,
)
from .solver import (
    HybridStep as HybridStep,
)
from .solver import (
    ProximalGradient as ProximalGradient,
)
from .solver import (
    WorkingSetPolicy as WorkingSetPolicy,
)

__version__ = "0.3.0"
__all__ = [
    "Binomial",
    "ChainFusedLasso",
    "ClusteredLasso",
    "Gaussian",
    "Hybrid",
    "HybridFit",
    "HybridStep",
    "Lasso",
    "LassoFit",
    "Normalization",
    "Preprocessing",
    "Problem",
    "ProximalFit",
    "ProximalGradient",
    "ScreeningDiagnostics",
    "Slope",
    "StoppingCriterion",
    "Termination",
    "WorkingSetDiagnostics",
    "WorkingSetPolicy",
    "__version__",
]
