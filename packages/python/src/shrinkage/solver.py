"""Explicit solver configurations with Rust's defaults and work budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import _shrinkage
from ._common import StoppingCriterion, stopping_spec

__all__ = ["Hybrid", "HybridStep", "ProximalGradient", "WorkingSetPolicy"]


@dataclass(frozen=True)
class HybridStep:
    """Curvature policy for hybrid proximal-gradient transitions.

    Use ``backtracking`` or ``fixed_bound``. A fixed bound uses the design's
    Frobenius bound. Backtracking doubles curvature after rejected trials.
    Logistic outer step halvings have a separate budget on :class:`Hybrid`.
    """

    kind: Literal["backtracking", "fixed_bound"]
    initial_curvature: float | None = None
    max_backtracks: int = 100

    @classmethod
    def backtracking(
        cls, *, initial_curvature: float | None = None, max_backtracks: int = 100
    ) -> HybridStep:
        """Use a positive seed or the largest averaged column curvature."""
        return cls("backtracking", initial_curvature, max_backtracks)

    @classmethod
    def fixed_bound(cls) -> HybridStep:
        """Use a computed global curvature bound without a spectral estimate."""
        return cls("fixed_bound")


@dataclass(frozen=True)
class WorkingSetPolicy:
    """Heuristic feature selection, independent of certified zero screening.

    Every policy retains current nonzeros. Excluded zeros may reenter after
    full checks. Only the original full-space mapping certifies convergence.
    """

    kind: Literal["auto", "disabled", "ever_active", "adaptive"]
    initial_size: int = 10
    prune: bool = True

    @classmethod
    def auto(cls) -> WorkingSetPolicy:
        """Use adaptive sets for single fits, starting with ten features."""
        return cls("auto")

    @classmethod
    def disabled(cls) -> WorkingSetPolicy:
        """Update every feature allowed by independently configured screening."""
        return cls("disabled")

    @classmethod
    def ever_active(cls) -> WorkingSetPolicy:
        """Retain features activated at accepted iterates."""
        return cls("ever_active")

    @classmethod
    def adaptive(
        cls, initial_size: int = 10, *, prune: bool = True
    ) -> WorkingSetPolicy:
        """Rank violations, growing the set and optionally pruning it."""
        return cls("adaptive", initial_size, prune)


@dataclass(frozen=True)
class Hybrid:
    """Hybrid PG and exact block sweeps, with IRLS for binomial models.

    Start with a PG transition, then alternate ``cluster_sweeps`` block sweeps
    with PG. ``max_iterations`` counts committed PG transitions plus sweeps;
    logistic fits share it across quadratic subproblems. Outer iterations
    count accepted IRLS steps. Inner iterations limit each quadratic model.
    Gaussian fitting ignores the outer and inner budgets.

    Checks run initially, every ``check_every`` iterations, and at the budget.
    Only the full original-loss mapping certifies convergence. ``stopping=None``
    uses absolute mapping tolerance ``1e-6``. Gap and KKT criteria are rejected.
    Screening and working sets are enabled independently by default.
    """

    stopping: StoppingCriterion | None = field(default=None, kw_only=True)
    max_iterations: int = field(default=10_000, kw_only=True)
    max_outer_iterations: int = field(default=100, kw_only=True)
    max_inner_iterations: int = field(default=1_000, kw_only=True)
    max_outer_backtracks: int = field(default=100, kw_only=True)
    cluster_sweeps: int = field(default=5, kw_only=True)
    check_every: int = field(default=10, kw_only=True)
    screening: bool = field(default=True, kw_only=True)
    working_set_policy: WorkingSetPolicy = field(
        default_factory=WorkingSetPolicy.auto, kw_only=True
    )
    step_policy: HybridStep = field(
        default_factory=HybridStep.backtracking, kw_only=True
    )

    def _native(self) -> _shrinkage._Solver:
        policy, step = self.working_set_policy, self.step_policy
        return _shrinkage._Solver.hybrid(
            stopping_spec(self.stopping),
            self.max_iterations,
            self.max_outer_iterations,
            self.max_inner_iterations,
            self.max_outer_backtracks,
            self.cluster_sweeps,
            self.check_every,
            self.screening,
            (policy.kind, policy.initial_size, policy.prune),
            (step.kind, step.initial_curvature, step.max_backtracks),
        )


@dataclass(frozen=True)
class ProximalGradient:
    """Backtracking PG on the original Gaussian or binomial objective.

    ``max_iterations`` limits accepted proximal updates. ``initial_step`` is
    finite and positive, and ``max_backtracks`` limits step halvings per
    iteration, with zero permitting only the first trial. ``stopping=None``
    uses absolute full-space mapping tolerance ``1e-6``. Gap and KKT criteria
    are rejected. Screening is enabled by default; heuristic working sets
    are available through :class:`Hybrid`.
    """

    stopping: StoppingCriterion | None = field(default=None, kw_only=True)
    max_iterations: int = field(default=10_000, kw_only=True)
    initial_step: float = field(default=1.0, kw_only=True)
    max_backtracks: int = field(default=100, kw_only=True)
    screening: bool = field(default=True, kw_only=True)

    def _native(self) -> _shrinkage._Solver:
        return _shrinkage._Solver.proximal_gradient(
            stopping_spec(self.stopping),
            self.max_iterations,
            self.initial_step,
            self.max_backtracks,
            self.screening,
        )
