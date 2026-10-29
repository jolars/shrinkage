"""Compose dense data, a datafit, a complete penalty, and an explicit solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

from ._common import Family, Normalization
from .datafit import Binomial, Gaussian
from .fit import HybridFit, ProximalFit
from .penalty import ChainFusedLasso, ClusteredLasso, Slope
from .solver import Hybrid, ProximalGradient

__all__ = ["Problem"]


@dataclass(frozen=True, eq=False)
class Problem:
    """A dense single-response problem, fitted with an explicit solver.

    ``x`` and the datafit's response are retained by reference until fitting.
    Each ``fit_with`` call independently copies them into Rust-owned storage.
    The input contract and normalization presets match :class:`shrinkage.Lasso`.
    Penalties act on optimization-scale coefficients. Returned coefficients,
    intercepts, and predictions use the original input scale.
    """

    x: object = field(repr=False)
    datafit: Gaussian | Binomial
    penalty: ClusteredLasso | ChainFusedLasso | Slope
    normalization: Normalization = field(default="auto", kw_only=True)

    @overload
    def fit_with(self, solver: Hybrid, *, copy: bool = True) -> HybridFit: ...

    @overload
    def fit_with(
        self, solver: ProximalGradient, *, copy: bool = True
    ) -> ProximalFit: ...

    def fit_with(
        self, solver: Hybrid | ProximalGradient, *, copy: bool = True
    ) -> HybridFit | ProximalFit:
        """Fit with Rust, releasing the GIL after copying all inputs.

        Inputs must remain unchanged until this call returns. ``copy=False``
        is unsupported. Invalid inputs or incompatible stopping rules raise
        ``TypeError`` or ``ValueError``. Nonfinite arithmetic raises
        ``FloatingPointError``, and line-search failures raise ``RuntimeError``.
        Iteration limits return finite results with ``termination`` equal to
        ``"iteration_limit"``. Results retain no problem or training arrays.
        """
        if not isinstance(self.datafit, Gaussian | Binomial):
            raise TypeError("datafit must be Gaussian or Binomial")
        if not isinstance(self.penalty, ClusteredLasso | ChainFusedLasso | Slope):
            raise TypeError("penalty must be ClusteredLasso, ChainFusedLasso, or Slope")
        if not isinstance(solver, Hybrid | ProximalGradient):
            raise TypeError("solver must be Hybrid or ProximalGradient")
        family: Family = (
            "gaussian" if isinstance(self.datafit, Gaussian) else "binomial"
        )
        native = solver._native().fit(
            self.x,
            self.datafit.y,
            family,
            self.datafit.fit_intercept,
            self.normalization,
            self.penalty._spec(),
            copy=copy,
        )
        return HybridFit(native) if isinstance(solver, Hybrid) else ProximalFit(native)
