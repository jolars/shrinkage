"""Complete penalties on optimization-scale coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["ChainFusedLasso", "ClusteredLasso", "Slope"]

PenaltySpec = tuple[str, float, float, tuple[float, ...] | None]


def _weights(weights: npt.ArrayLike) -> tuple[float, ...]:
    values = np.asarray(weights)
    if values.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if values.dtype.kind not in "fiu":
        raise TypeError("weights must contain real numbers")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class ClusteredLasso:
    """L1 plus uniform all-pairs fusion of signed coefficient values.

    The penalty is ``l1 * sum(abs(theta)) + fusion * sum(i<j,
    abs(theta[i] - theta[j]))``. Strengths must be finite and nonnegative.
    Each unordered pair contributes once. The intercept is excluded.
    """

    l1: float
    fusion: float

    def _spec(self) -> PenaltySpec:
        return ("clustered_lasso", self.l1, self.fusion, None)


@dataclass(frozen=True, init=False)
class ChainFusedLasso:
    """L1 plus weighted fusion of adjacent coefficients in feature order.

    The penalty is ``l1 * sum(abs(theta)) + fusion * sum(weights *
    abs(diff(theta)))``. Strengths and weights must be finite and nonnegative.
    ``weights=None`` uses unit weights; explicit weights must have length
    ``max(n_features - 1, 0)``. Zero weights cut the chain. Weight sequences
    are copied into an immutable tuple. The intercept is excluded.
    """

    l1: float
    fusion: float
    weights: tuple[float, ...] | None

    def __init__(
        self, l1: float, fusion: float, *, weights: npt.ArrayLike | None = None
    ) -> None:
        object.__setattr__(self, "l1", l1)
        object.__setattr__(self, "fusion", fusion)
        object.__setattr__(
            self, "weights", None if weights is None else _weights(weights)
        )

    def _spec(self) -> PenaltySpec:
        return ("chain_fused_lasso", self.l1, self.fusion, self.weights)


@dataclass(frozen=True, init=False)
class Slope:
    """Sorted L1 penalty with complete rank weights, including strength.

    Supply one finite, nonnegative, nonincreasing weight per feature. Weights
    multiply coefficients sorted by decreasing magnitude. Equal weights give
    L1, and zero weights give no penalty. The supplied sequence is copied into
    an immutable tuple. The intercept is excluded.
    """

    weights: tuple[float, ...]

    def __init__(self, weights: npt.ArrayLike) -> None:
        object.__setattr__(self, "weights", _weights(weights))

    def _spec(self) -> PenaltySpec:
        return ("slope", 0.0, 0.0, self.weights)
