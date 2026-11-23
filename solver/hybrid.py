"""Hybrid solver dispatcher and session runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.game import GameSession

from .constraints import count_slot_candidates
from .exact_solver import ExactPoolSolver
from .factor_solver import FactorProbeSolver


class SolverProtocol(Protocol):
    def next_guess(self, session: GameSession) -> list[list[int]]: ...


class HybridSolver:
    """Automatically selects ExactPoolSolver or FactorProbeSolver by slot space size."""

    def __init__(
        self,
        seed: int = 0,
        diversity: float = 0.0,
        exact_threshold: int = 2000,
    ) -> None:
        self.exact_threshold = exact_threshold
        self.exact_solver = ExactPoolSolver(seed=seed, diversity=diversity)
        self.factor_solver = FactorProbeSolver(seed=seed + 100_003, diversity=diversity)

    def select_backend(self, session: GameSession) -> tuple[str, SolverProtocol]:
        if count_slot_candidates(session.config) <= self.exact_threshold:
            return "exact", self.exact_solver
        return "factor", self.factor_solver

    def next_guess(self, session: GameSession) -> list[list[int]]:
        _, backend = self.select_backend(session)
        return backend.next_guess(session)


@dataclass(frozen=True)
class SolveResult:
    won: bool
    turns: int
    backend: str


def solve_session(
    session: GameSession,
    solver: SolverProtocol | None = None,
    max_steps: int = 300,
) -> SolveResult:
    """Play a GameSession to completion (or up to `max_steps`)."""
    active_solver = solver if solver is not None else HybridSolver()
    backend_name = (
        active_solver.select_backend(session)[0]
        if isinstance(active_solver, HybridSolver)
        else type(active_solver).__name__
    )

    steps = 0
    while session.status == "playing" and steps < max_steps:
        guess = active_solver.next_guess(session)
        session.submit_guess(guess)
        steps += 1

    return SolveResult(
        won=(session.status == "won"),
        turns=len(session.history),
        backend=backend_name,
    )
