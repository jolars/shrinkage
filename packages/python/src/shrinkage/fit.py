"""Owned parameters and diagnostics for hybrid and proximal-gradient fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import _shrinkage
from ._common import (
    Family,
    FloatArray,
    Preprocessing,
    StoppingCriterion,
    Termination,
    preprocessing,
)
from .solver import HybridStep, WorkingSetPolicy

__all__ = ["HybridFit", "ProximalFit", "ScreeningDiagnostics", "WorkingSetDiagnostics"]


@dataclass(frozen=True)
class ScreeningDiagnostics:
    """Work and certified zero restrictions for the original problem.

    ``supported`` reports whether screening was enabled and supported.
    ``active_dimension`` counts features remaining after safe screening,
    independently of any heuristic working-set exclusions.
    """

    supported: bool
    certificate_passes: int
    zero_tests: int
    screened_features: int
    active_dimension: int
    rebuilds: int
    fallback_certificates: int


@dataclass(frozen=True)
class WorkingSetDiagnostics:
    """Heuristic feature-selection work, separate from safe screening.

    ``policy`` is resolved rather than ``auto``. Full checks include logistic
    quadratic subproblem checks. Feature and block visits count numerical
    work, not the number of nonzero coefficients.
    """

    policy: WorkingSetPolicy
    full_checks: int
    restricted_checks: int
    rebuilds: int
    expansions: int
    reactivations: int
    active_dimension: int
    peak_active_dimension: int
    feature_visits: int
    block_visits: int


@dataclass(frozen=True, eq=False)
class ProximalFit:
    """Owned original-scale parameters and full original-loss diagnostics.

    Inspect ``termination`` before treating the result as converged. Arrays
    are independent snapshots, and the result retains no training data. The
    objective, mapping, and dual diagnostics use optimization-scale penalties.
    """

    _native: _shrinkage._ModelFit = field(repr=False)

    @property
    def solver(self) -> Literal["hybrid", "proximal_gradient"]:
        """Explicitly selected numerical method."""
        return self._native.solver

    @property
    def family(self) -> Family:
        """``"gaussian"`` or ``"binomial"``; binomial predictions are logits."""
        return self._native.family

    @property
    def coefficients(self) -> FloatArray:
        """Original-scale coefficients, shape ``(n_features,)``."""
        return np.asarray(self._native.coefficients, dtype=np.float64)

    @property
    def intercept(self) -> float:
        """Original-scale intercept, including the centering correction."""
        return self._native.intercept

    @property
    def preprocessing(self) -> Preprocessing:
        """Training transformations; prediction never estimates new ones."""
        return preprocessing(self._native)

    @property
    def termination(self) -> Termination:
        """Whether the full mapping passed or an iteration budget expired."""
        return self._native.termination

    @property
    def iterations(self) -> int:
        """Accepted PG updates, or hybrid PG transitions plus block sweeps."""
        return self._native.iterations

    @property
    def objective(self) -> float:
        """Original averaged loss plus the complete optimization-scale penalty."""
        return self._native.objective

    @property
    def stopping(self) -> StoppingCriterion:
        """Resolved full-space proximal-gradient mapping criterion."""
        return StoppingCriterion.proximal_gradient_mapping(self.stopping_threshold)

    @property
    def stopping_value(self) -> float:
        """Full mapping infinity norm, including the free-intercept derivative."""
        return self._native.stopping_value

    @property
    def stopping_threshold(self) -> float:
        """Absolute mapping tolerance used to certify convergence."""
        return self._native.stopping_threshold

    @property
    def step_size(self) -> float:
        """Step used to evaluate the final full original-loss mapping."""
        return self._native.step_size

    @property
    def dual_certificate(self) -> FloatArray | None:
        """Feasible original-loss dual vector, shape ``(n_samples,)``, if available."""
        values = self._native.dual_certificate
        return None if values is None else np.asarray(values, dtype=np.float64)

    @property
    def dual_objective(self) -> float | None:
        """Original-problem dual lower bound, or ``None`` without screening."""
        return self._native.dual_objective

    @property
    def duality_gap(self) -> float | None:
        """Padded original-problem gap; the full mapping controls convergence."""
        return self._native.duality_gap

    @property
    def screening_diagnostics(self) -> ScreeningDiagnostics:
        """Certified restrictions and work in the original feature space."""
        return ScreeningDiagnostics(*self._native.screening_diagnostics)

    def predict(self, x: object, *, copy: bool = True) -> FloatArray:
        """Predict means or logits, returning float64 shape ``(n_samples,)``.

        Binomial predictions are logits, not probabilities or class labels.
        Input rules match fitting, except zero rows are allowed. Inputs are
        copied before releasing the GIL and must remain unchanged until return.
        Prediction uses original-scale coefficients without refitting any
        preprocessing. ``copy=False`` is unsupported.
        """
        return np.asarray(self._native.predict(x, copy=copy), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class HybridFit(ProximalFit):
    """A hybrid result with optimization-scale clusters and inner/outer work.

    Clustered lasso groups signed values in increasing order. Chain fused
    lasso groups contiguous segments in feature order, retaining cuts at zero
    weights. SLOPE groups decreasing nonnegative magnitudes and retains each
    feature's sign. Training scales can change equalities among the reported
    original-scale coefficients.
    """

    @property
    def cluster_labels(self) -> npt.NDArray[np.intp]:
        """Zero-based indices into ``cluster_values``, shape ``(n_features,)``."""
        return np.asarray(self._native.cluster_labels, dtype=np.intp)

    @property
    def cluster_values(self) -> FloatArray:
        """Optimization-scale block values, shape ``(n_clusters,)``."""
        return np.asarray(self._native.cluster_values, dtype=np.float64)

    @property
    def coefficient_signs(self) -> FloatArray:
        """Optimization-scale feature signs (-1, 0, or 1), shape ``(n_features,)``."""
        return np.asarray(self._native.coefficient_signs, dtype=np.float64)

    @property
    def pg_steps(self) -> int:
        """Committed PG transitions, excluding diagnostic proximal calls."""
        return self._native.pg_steps

    @property
    def coordinate_sweeps(self) -> int:
        """Complete block-coordinate sweeps over the current working set."""
        return self._native.coordinate_sweeps

    @property
    def diagnostic_checks(self) -> int:
        """Refreshed inner checks, including quadratic checks in logistic fits."""
        return self._native.diagnostic_checks

    @property
    def backtracks(self) -> int:
        """Rejected PG trials followed by curvature doubling."""
        return self._native.backtracks

    @property
    def outer_iterations(self) -> int:
        """Accepted logistic IRLS steps; zero for Gaussian fits."""
        return self._native.outer_iterations

    @property
    def outer_backtracks(self) -> int:
        """Rejected logistic trials followed by step halving; zero for Gaussian."""
        return self._native.outer_backtracks

    @property
    def step_policy(self) -> HybridStep:
        """Configured Gaussian or logistic quadratic curvature policy."""
        kind, curvature, backtracks = self._native.step_policy
        return HybridStep(kind, curvature, backtracks)

    @property
    def working_set_diagnostics(self) -> WorkingSetDiagnostics:
        """Effective working-set policy and feature and block work counters."""
        policy, *counts = self._native.working_set_diagnostics
        return WorkingSetDiagnostics(WorkingSetPolicy(*policy), *counts)
