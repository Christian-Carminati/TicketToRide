"""Opponent policies for the Ticket to Ride reinforcement learning environments."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from best_solution.heuristic_solver import HeuristicSolver

if False:  # pragma: no cover - for type checking without circular import
    from rl.envs import SingleAgentTicketToRideEnv


class OpponentPolicy(ABC):
    """Base class implemented by scripted opponents."""

    def reset(self, env: "SingleAgentTicketToRideEnv") -> None:
        """Called when the environment is reset."""

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        env: "SingleAgentTicketToRideEnv",
    ) -> Optional[int]:
        """Return the index of the action to execute or ``None`` to skip the move."""


class RandomOpponentPolicy(OpponentPolicy):
    """Opponent that samples uniformly from the valid actions."""

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        env: "SingleAgentTicketToRideEnv",
    ) -> Optional[int]:
        if not valid_actions:
            return None
        return random.choice(valid_actions)


class GreedyRouteOpponentPolicy(OpponentPolicy):
    """Opponent that chooses the available route with the highest score."""

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        env: "SingleAgentTicketToRideEnv",
    ) -> Optional[int]:
        if not valid_actions:
            return None
        num_routes = getattr(env, "num_routes", len(getattr(env, "routes", [])))
        routes = getattr(env, "routes", [])
        # Consider only route-claim actions for greedy choice
        route_actions = [a for a in valid_actions if isinstance(a, int) and 0 <= a < num_routes]
        if route_actions:
            return max(route_actions, key=lambda idx: getattr(routes[idx], "length", 0))
        # Fallback: prefer drawing 2 from deck if encoded as first card action
        draw_two_from_deck = num_routes  # by spec in FullGameSingleAgentEnv
        if draw_two_from_deck in valid_actions:
            return draw_two_from_deck
        # Otherwise pick any valid action
        return random.choice(list(valid_actions))


class HeuristicOpponentPolicy(OpponentPolicy):
    """Opponent that pre-computes a target set of routes using a heuristic solver."""

    def __init__(self, solver: HeuristicSolver, strategy: str = "greedy") -> None:
        self.solver = solver
        self.strategy = strategy
        self._target_route_pairs: list[tuple[str, str]] = []

    def reset(self, env: "SingleAgentTicketToRideEnv") -> None:
        self._target_route_pairs = []
        if self.strategy == "greedy":
            solution = self.solver.greedy_solve()
        elif self.strategy == "tabu":
            solution = self.solver.tabu_search_solve(max_iterations=200)
        elif self.strategy == "genetic":
            solution = self.solver.genetic_algorithm_solve(population_size=20, generations=30)
        else:
            solution = self.solver.greedy_solve()
        self._target_route_pairs = [tuple(sorted((route[0], route[1]))) for route in solution.routes]

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        env: "SingleAgentTicketToRideEnv",
    ) -> Optional[int]:
        if not valid_actions:
            return None

        # Try to follow the pre-computed plan first.
        num_routes = getattr(env, "num_routes", len(getattr(env, "routes", [])))
        routes = getattr(env, "routes", [])
        route_actions = [a for a in valid_actions if isinstance(a, int) and 0 <= a < num_routes]
        if route_actions:
            for pair in self._target_route_pairs:
                for idx in route_actions:
                    route = routes[idx]
                    cities = tuple(sorted((route.city1, route.city2)))
                    if cities == pair:
                        return idx
        # Fallback to greedy selection by length.
        if route_actions:
            return max(route_actions, key=lambda idx: getattr(routes[idx], "length", 0))
        # Prefer drawing 2 from deck if available
        draw_two_from_deck = num_routes
        if draw_two_from_deck in valid_actions:
            return draw_two_from_deck
        return random.choice(list(valid_actions))


def build_opponent(name: str, *, graph, tickets_file: str) -> OpponentPolicy:
    """Factory returning an opponent policy by name."""

    name = name.lower()
    if name == "random":
        return RandomOpponentPolicy()
    if name == "greedy":
        return GreedyRouteOpponentPolicy()
    if name in {"heuristic", "greedy_heuristic", "tabu", "genetic"}:
        solver = HeuristicSolver(graph, tickets_file)
        strategy = {
            "heuristic": "greedy",
            "greedy_heuristic": "greedy",
            "tabu": "tabu",
            "genetic": "genetic",
        }[name]
        return HeuristicOpponentPolicy(solver, strategy=strategy)
    raise ValueError(f"Unknown opponent policy '{name}'")


__all__ = [
    "OpponentPolicy",
    "RandomOpponentPolicy",
    "GreedyRouteOpponentPolicy",
    "HeuristicOpponentPolicy",
    "build_opponent",
]
