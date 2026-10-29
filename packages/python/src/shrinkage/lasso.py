"""Dense, single-response Gaussian lasso backed by Rust coordinate descent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _shrinkage
from ._common import (
    FloatArray,
    preprocessing,
    stopping_spec,
)
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

__all__ = [
    "Lasso",
    "LassoFit",
    "Normalization",
    "Preprocessing",
    "StoppingCriterion",
    "Termination",
]


@dataclass(frozen=True)
class Lasso:
    """Configure Gaussian lasso, then call :meth:`fit` for an owned result.

    Minimize ``||y - X_tilde @ theta - b||**2 / (2*n) + lambda_ * ||theta||_1``.
    The intercept is unpenalized. ``lambda_`` must be finite and nonnegative.
    ``normalization="auto"`` divides columns by population standard deviations
    and centers by their means when ``fit_intercept=True``. Other presets are
    independent of the intercept setting. Penalties act on normalized
    coefficients; results and predictions use the original input scale.

    By default, positive penalties use relative duality gap ``1e-6`` and zero
    penalty uses absolute KKT violation ``1e-6``. ``stopping`` overrides this
    choice. ``max_iterations`` limits complete coordinate sweeps and must be
    positive. Options are validated when fitting.
    """

    lambda_: float
    fit_intercept: bool = field(default=True, kw_only=True)
    normalization: Normalization = field(default="auto", kw_only=True)
    stopping: StoppingCriterion | None = field(default=None, kw_only=True)
    max_iterations: int = field(default=10_000, kw_only=True)

    def fit(self, x: object, y: object, *, copy: bool = True) -> LassoFit:
        """Fit exact NumPy float64 arrays of shapes ``(n, p)`` and ``(n,)``.

        Strided, writable, and read-only arrays are supported. Both inputs
        are copied into Rust-owned storage with the GIL held. Callers must
        prevent concurrent mutation or resizing until copying finishes.
        Numerical fitting releases the GIL and calls no Python callbacks.
        ``copy=False`` raises ``ValueError``.

        Invalid data or options raise ``TypeError`` or ``ValueError``.
        Nonfinite arithmetic from finite inputs raises ``FloatingPointError``.
        An exhausted budget returns a fit with termination ``"iteration_limit"``.
        """
        native = _shrinkage._Lasso(
            self.lambda_,
            self.fit_intercept,
            self.normalization,
            stopping_spec(self.stopping),
            self.max_iterations,
        )
        return LassoFit(native.fit(x, y, copy=copy))


@dataclass(frozen=True, eq=False)
class LassoFit:
    """Owned lasso parameters, fitted preprocessing, and final diagnostics.

    Check ``termination`` before treating a fit as converged. Properties are
    read-only, and returned arrays are independent copies. The result retains
    neither the training arrays nor their Rust snapshots.
    """

    _native: _shrinkage._LassoFit = field(repr=False)

    @property
    def coefficients(self) -> FloatArray:
        """Original-scale coefficients with shape ``(n_features,)``."""
        return np.asarray(self._native.coefficients, dtype=np.float64)

    @property
    def intercept(self) -> float:
        """Original-scale intercept, including the correction for centering."""
        return self._native.intercept

    @property
    def preprocessing(self) -> Preprocessing:
        """Training transformations; prediction never estimates new ones."""
        return preprocessing(self._native)

    @property
    def termination(self) -> Termination:
        """``"converged"`` or ``"iteration_limit"`` from the Rust solver."""
        return self._native.termination

    @property
    def iterations(self) -> int:
        """Complete coordinate sweeps; initially optimal fits take zero."""
        return self._native.iterations

    @property
    def objective(self) -> float:
        """Final average squared loss plus the optimization-scale L1 penalty."""
        return self._native.objective

    @property
    def kkt_violation(self) -> float:
        """Maximum KKT violation, including the free intercept derivative."""
        return self._native.kkt_violation

    @property
    def dual_certificate(self) -> FloatArray:
        """Feasible Gaussian dual vector with shape ``(n_samples,)``."""
        return np.asarray(self._native.dual_certificate, dtype=np.float64)

    @property
    def dual_objective(self) -> float:
        """Dual objective ``y @ u - n * ||u||**2 / 2``."""
        return self._native.dual_objective

    @property
    def duality_gap(self) -> float:
        """Nonnegative primal minus dual; it can be loose at zero penalty."""
        return self._native.duality_gap

    @property
    def reference_loss(self) -> float:
        """Fixed average loss at zero coefficients and the optimal free intercept."""
        return self._native.reference_loss

    @property
    def stopping(self) -> StoppingCriterion:
        """Resolved criterion, including the default selected for this penalty."""
        kind, absolute, relative = self._native.stopping
        return StoppingCriterion(kind, absolute, relative)

    @property
    def stopping_value(self) -> float:
        """Final value of the selected convergence check on refreshed state."""
        return self._native.stopping_value

    @property
    def stopping_threshold(self) -> float:
        """Threshold applied to the selected convergence check."""
        return self._native.stopping_threshold

    def predict(self, x: object, *, copy: bool = True) -> FloatArray:
        """Predict raw NumPy float64 rows, returning shape ``(n_samples,)``.

        Use training-column order. The dense input contract also applies here,
        with empty batches allowed. Prediction copies inputs before releasing
        the GIL, uses original-scale parameters, and never refits preprocessing.
        Invalid shapes or values raise ``TypeError`` or ``ValueError``;
        nonfinite prediction arithmetic raises ``FloatingPointError``.
        """
        return np.asarray(self._native.predict(x, copy=copy), dtype=np.float64)
