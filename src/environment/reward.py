"""Configurable reward calculation engines."""

from abc import ABC, abstractmethod
from typing import Any

from src.game.state import GameState


class BaseRewardCalculator(ABC):
    """Abstract base class for modular reward functions."""

    @abstractmethod
    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        """Compute the step reward for player_index."""


class DefaultRewardCalculator(BaseRewardCalculator):
    """Configurable reward calculation module (Reward V1)."""

    def __init__(
        self,
        route_points_weight: float = 1.0,
        ticket_completion_weight: float = 10.0,
        win_bonus: float = 20.0,
        loss_penalty: float = 10.0,
    ) -> None:
        self.route_points_weight = route_points_weight
        self.ticket_completion_weight = ticket_completion_weight
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty

    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        # Skeleton reward calculation - refined in Phase 3 & 7
        return 0.0
