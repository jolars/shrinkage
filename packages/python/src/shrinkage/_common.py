"""Shared configuration and preprocessing for owned numerical results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

Normalization = Literal[
    "auto", "none", "center", "standardize", "min_max", "max_abs", "l1", "l2"
]
Termination = Literal["converged", "iteration_limit"]
Family = Literal["gaussian", "binomial"]
FloatArray = npt.NDArray[np.float64]
StoppingSpec = tuple[str, float, float]


@dataclass(frozen=True)
class StoppingCriterion:
    """A convergence check with tolerances in its own units.

    Gaussian lasso supports ``duality_gap`` and ``kkt_violation``. Hybrid and
    proximal gradient support only ``proximal_gradient_mapping``. KKT and
    mapping tolerances must be finite and strictly positive and use no
    relative component. Gap tolerances must be finite and nonnegative, with
    at least one positive component. Compatibility is checked when fitting.
    """

    kind: Literal["duality_gap", "kkt_violation", "proximal_gradient_mapping"]
    absolute: float
    relative: float = 0.0

    @classmethod
    def duality_gap(
        cls, relative: float = 1e-6, *, absolute: float = 0.0
    ) -> StoppingCriterion:
        """Require gap <= absolute + relative * fixed null-model loss."""
        return cls("duality_gap", absolute, relative)

    @classmethod
    def kkt_violation(cls, absolute: float = 1e-6) -> StoppingCriterion:
        """Require the largest absolute KKT violation <= absolute."""
        return cls("kkt_violation", absolute)

    @classmethod
    def proximal_gradient_mapping(cls, absolute: float = 1e-6) -> StoppingCriterion:
        """Bound the full mapping and the free-intercept derivative."""
        return cls("proximal_gradient_mapping", absolute)


def stopping_spec(stopping: StoppingCriterion | None) -> StoppingSpec | None:
    return (
        None
        if stopping is None
        else (stopping.kind, stopping.absolute, stopping.relative)
    )


@dataclass(frozen=True, eq=False)
class Preprocessing:
    """Training-column normalization resolved during fitting.

    ``centers`` and ``scales`` have shape ``(n_features,)`` when enabled and
    are ``None`` otherwise. Zero computed scales become one. Arrays are
    independent snapshots; changing them cannot change a fitted model.
    """

    centering: str
    scaling: str
    centers: FloatArray | None
    scales: FloatArray | None


class _PreprocessedFit(Protocol):
    @property
    def centers(self) -> list[float] | None: ...
    @property
    def scales(self) -> list[float] | None: ...
    @property
    def centering(self) -> str: ...
    @property
    def scaling(self) -> str: ...


def preprocessing(native: _PreprocessedFit) -> Preprocessing:
    centers, scales = native.centers, native.scales
    return Preprocessing(
        native.centering,
        native.scaling,
        None if centers is None else np.asarray(centers, dtype=np.float64),
        None if scales is None else np.asarray(scales, dtype=np.float64),
    )
