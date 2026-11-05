"""Warm regularization paths with complete tuning settings and owned fits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

import numpy as np

from . import _shrinkage
from ._common import Family, FloatArray
from .fit import ProximalFit
from .penalty import ChainFusedLasso, ClusteredLasso, PenaltySpec, Slope

__all__ = ["PathFit", "PathOptions", "PathTermination"]

PathTermination = Literal["completed", "deviance_change", "deviance_ratio"]
FitT = TypeVar("FitT", bound=ProximalFit)
OptionsSpec = tuple[int, float | None, float | None, bool, int, float, float]


@dataclass(frozen=True)
class PathOptions:
    """Generate a decreasing log-spaced sequence on the optimization scale.

    ``nlambda`` and ``min_points`` must be positive. The endpoint ratio is in
    ``(0, 1]`` and defaults to ``0.01`` when samples are fewer than features,
    or ``0.0001`` otherwise. ``lambda_start`` overrides the anchor without
    changing the reported true zero threshold. A zero anchor gives all zeros.

    Early stopping requires converged points and at least ``min_points`` fits.
    The change tolerance is finite and nonnegative; zero disables that check.
    The explained null-deviance tolerance is in ``(0, 1]``. Disable early
    stopping to fit every requested point. Options are validated when fitting.
    """

    nlambda: int = field(default=100, kw_only=True)
    lambda_min_ratio: float | None = field(default=None, kw_only=True)
    lambda_start: float | None = field(default=None, kw_only=True)
    early_stopping: bool = field(default=True, kw_only=True)
    min_points: int = field(default=5, kw_only=True)
    deviance_change_tolerance: float = field(default=1e-5, kw_only=True)
    deviance_ratio_tolerance: float = field(default=0.999, kw_only=True)

    def _spec(self) -> OptionsSpec:
        return (
            self.nlambda,
            self.lambda_min_ratio,
            self.lambda_start,
            self.early_stopping,
            self.min_points,
            self.deviance_change_tolerance,
            self.deviance_ratio_tolerance,
        )


def _penalty(spec: PenaltySpec) -> ClusteredLasso | ChainFusedLasso | Slope:
    name, l1, fusion, weights = spec
    if name == "clustered_lasso":
        return ClusteredLasso(l1, fusion)
    if name == "chain_fused_lasso":
        return ChainFusedLasso(l1, fusion, weights=weights)
    assert name == "slope" and weights is not None
    return Slope(weights)


@dataclass(frozen=True, eq=False)
class PathFit(Generic[FitT]):
    """Owned path in fitting order, with a complete result at every point.

    ``fits[i]`` exposes every single-fit diagnostic and prediction method.
    ``penalties[i]`` retains both fusion strengths and all edge or rank
    weights. Finite iteration-limit points remain in the path and warm-start
    the next point. A fitting error raises without returning a partial path.
    Results retain no training arrays. Returned arrays are independent copies.
    """

    _native: _shrinkage._ModelPath = field(repr=False)
    _wrap: Callable[[_shrinkage._ModelFit], FitT] = field(repr=False)

    def __len__(self) -> int:
        return self._native.npoints

    @property
    def family(self) -> Family:
        """Gaussian means or binomial logits, consistent across all points."""
        return self._native.family

    @property
    def solver(self) -> Literal["hybrid", "proximal_gradient"]:
        """Explicit numerical method used for every point, including empty paths."""
        return self._native.solver

    @property
    def n_features(self) -> int:
        """Training column count, retained even for an empty path."""
        return self._native.nfeatures

    @property
    def fits(self) -> tuple[FitT, ...]:
        """Per-point owned fits, aligned with ``penalties`` and parameter rows."""
        return tuple(self._wrap(fit) for fit in self._native.fits)

    @property
    def penalties(self) -> tuple[ClusteredLasso | ChainFusedLasso | Slope, ...]:
        """Complete immutable penalty settings for each fitted point."""
        return tuple(_penalty(spec) for spec in self._native.penalties)

    @property
    def coefficients(self) -> FloatArray:
        """Original-scale float64 coefficients, shape ``(n_points, n_features)``."""
        return np.asarray(self._native.coefficients, dtype=np.float64).reshape(
            len(self), self._native.nfeatures
        )

    @property
    def intercepts(self) -> FloatArray:
        """Original-scale float64 intercepts, shape ``(n_points,)``."""
        return np.asarray(self._native.intercepts, dtype=np.float64)

    @property
    def lambdas(self) -> FloatArray | None:
        """Automatic scalar multipliers, shape ``(n_points,)``; explicit: ``None``."""
        values = self._native.lambdas
        return None if values is None else np.asarray(values, dtype=np.float64)

    @property
    def maximum_penalty(self) -> float | None:
        """True zero threshold: finite or ``inf`` for automatic paths; else ``None``."""
        return self._native.maximum_penalty

    @property
    def lambda_start(self) -> float | None:
        """Used automatic anchor, including an override; explicit paths: ``None``."""
        return self._native.lambda_start

    @property
    def termination(self) -> PathTermination:
        """Path completion or early stopping, independent of point convergence."""
        return self._native.termination

    @property
    def null_deviance(self) -> float | None:
        """Averaged automatic-path null-model deviance; explicit paths: ``None``."""
        return self._native.null_deviance

    @property
    def deviances(self) -> FloatArray | None:
        """Automatic fitted deviances, shape ``(n_points,)``; explicit: ``None``."""
        values = self._native.deviances
        return None if values is None else np.asarray(values, dtype=np.float64)

    def predict(self, x: object, *, copy: bool = True) -> FloatArray:
        """Predict float64 means or logits, shape ``(n_samples, n_points)``.

        Columns follow path order, including duplicates. Empty batches and
        empty paths retain both axes. The dense input and copying rules match
        single fits. Inputs are copied once before Rust releases the GIL, and
        must remain unchanged until return. Prediction never refits transforms.
        """
        nrows, values = self._native.predict(x, copy=copy)
        return np.asarray(values, dtype=np.float64).reshape(len(self), nrows).T
