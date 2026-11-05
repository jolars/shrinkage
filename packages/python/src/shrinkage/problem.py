"""Compose dense data, a datafit, a complete penalty, and an explicit solver."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import overload

from ._common import Family, Normalization
from .datafit import Binomial, Gaussian
from .fit import HybridFit, ProximalFit
from .path import PathFit, PathOptions
from .penalty import ChainFusedLasso, ClusteredLasso, Slope
from .solver import HybridCoordinateDescent, ProximalGradient

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

    def _validate(self, solver: HybridCoordinateDescent | ProximalGradient) -> Family:
        if not isinstance(self.datafit, Gaussian | Binomial):
            raise TypeError("datafit must be Gaussian or Binomial")
        if not isinstance(self.penalty, ClusteredLasso | ChainFusedLasso | Slope):
            raise TypeError("penalty must be ClusteredLasso, ChainFusedLasso, or Slope")
        if not isinstance(solver, HybridCoordinateDescent | ProximalGradient):
            raise TypeError(
                "solver must be HybridCoordinateDescent or ProximalGradient"
            )
        return "gaussian" if isinstance(self.datafit, Gaussian) else "binomial"

    @overload
    def fit_with(
        self, solver: HybridCoordinateDescent, *, copy: bool = True
    ) -> HybridFit: ...

    @overload
    def fit_with(
        self, solver: ProximalGradient, *, copy: bool = True
    ) -> ProximalFit: ...

    def fit_with(
        self, solver: HybridCoordinateDescent | ProximalGradient, *, copy: bool = True
    ) -> HybridFit | ProximalFit:
        """Fit with Rust, releasing the GIL after copying all inputs.

        Inputs must remain unchanged until this call returns. ``copy=False``
        is unsupported. Invalid inputs or incompatible stopping rules raise
        ``TypeError`` or ``ValueError``. Nonfinite arithmetic raises
        ``FloatingPointError``, and line-search failures raise ``RuntimeError``.
        Iteration limits return finite results with ``termination`` equal to
        ``"iteration_limit"``. Results retain no problem or training arrays.
        """
        family = self._validate(solver)
        native = solver._native().fit(
            self.x,
            self.datafit.y,
            family,
            self.datafit.fit_intercept,
            self.normalization,
            self.penalty._spec(),
            copy=copy,
        )
        return (
            HybridFit(native)
            if isinstance(solver, HybridCoordinateDescent)
            else ProximalFit(native)
        )

    @overload
    def fit_path_with(
        self,
        solver: HybridCoordinateDescent,
        penalties: Iterable[ClusteredLasso | ChainFusedLasso | Slope],
        *,
        copy: bool = True,
    ) -> PathFit[HybridFit]: ...

    @overload
    def fit_path_with(
        self,
        solver: ProximalGradient,
        penalties: Iterable[ClusteredLasso | ChainFusedLasso | Slope],
        *,
        copy: bool = True,
    ) -> PathFit[ProximalFit]: ...

    def fit_path_with(
        self,
        solver: HybridCoordinateDescent | ProximalGradient,
        penalties: Iterable[ClusteredLasso | ChainFusedLasso | Slope],
        *,
        copy: bool = True,
    ) -> PathFit[HybridFit] | PathFit[ProximalFit]:
        """Fit complete penalties in caller order with Rust workspace reuse.

        All points must match the problem penalty's type. Both strengths and
        weights may change independently. Empty, repeated, and nonmonotone
        sequences are accepted. The problem's own penalty supplies only the
        type for explicit paths. HybridCoordinateDescent ``auto`` working sets
        resolve to ``ever_active``. Each point has its own solver budget, and finite
        iteration-limit fits warm-start subsequent points. Errors raise
        without returning a partial path. Dense input and GIL rules match fit.
        """
        family = self._validate(solver)
        points = tuple(penalties)
        if any(type(point) is not type(self.penalty) for point in points):
            raise TypeError(
                "path penalties must have the same type as the problem penalty"
            )
        native = solver._native().path(
            self.x,
            self.datafit.y,
            family,
            self.datafit.fit_intercept,
            self.normalization,
            self.penalty._spec(),
            penalties=[point._spec() for point in points],
            copy=copy,
        )
        return (
            PathFit(native, HybridFit)
            if isinstance(solver, HybridCoordinateDescent)
            else PathFit(native, ProximalFit)
        )

    @overload
    def fit_regularization_path_with(
        self,
        solver: HybridCoordinateDescent,
        options: PathOptions | None = None,
        *,
        copy: bool = True,
    ) -> PathFit[HybridFit]: ...

    @overload
    def fit_regularization_path_with(
        self,
        solver: ProximalGradient,
        options: PathOptions | None = None,
        *,
        copy: bool = True,
    ) -> PathFit[ProximalFit]: ...

    def fit_regularization_path_with(
        self,
        solver: HybridCoordinateDescent | ProximalGradient,
        options: PathOptions | None = None,
        *,
        copy: bool = True,
    ) -> PathFit[HybridFit] | PathFit[ProximalFit]:
        """Generate and warm-fit a decreasing sequence from Rust's null model.

        Scalar strengths multiply both clustered or chain strengths, or all
        SLOPE rank weights. Chain edge weights stay fixed. Paths with no finite
        zero threshold require an explicit ``lambda_start``. Free-intercept
        binomial paths reject all-zero or all-one responses because their
        null optimum is not finite. Inputs are copied once for the whole path.
        """
        family = self._validate(solver)
        if options is None:
            options = PathOptions()
        if not isinstance(options, PathOptions):
            raise TypeError("options must be PathOptions")
        native = solver._native().path(
            self.x,
            self.datafit.y,
            family,
            self.datafit.fit_intercept,
            self.normalization,
            self.penalty._spec(),
            options=options._spec(),
            copy=copy,
        )
        return (
            PathFit(native, HybridFit)
            if isinstance(solver, HybridCoordinateDescent)
            else PathFit(native, ProximalFit)
        )
