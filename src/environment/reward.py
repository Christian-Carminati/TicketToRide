"""Configurable reward calculation engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.graph import check_ticket_completed, check_tickets_completed_batch
from src.game.maps import load_usa_board
from src.game.rules import GameRules
from src.game.state import GameState


@dataclass
class RewardWeights:
    """Configurable hyperparameters for reward shaping and terminal outcomes."""

    route_points_weight: float = 1.0
    ticket_completion_weight: float = 1.0
    step_penalty: float = 0.0
    win_bonus: float = 20.0
    loss_penalty: float = 10.0
    score_diff_weight: float = 0.5
    ticket_failure_penalty_weight: float = 1.0


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
        weights: RewardWeights | None = None,
        board: Board | None = None,
    ) -> None:
        self.weights = weights or RewardWeights()
        if board is None:
            usa_board, _ = load_usa_board()
            self._routes_by_id = {r.id: r for r in usa_board.routes}
        else:
            self._routes_by_id = {r.id: r for r in board.routes}

    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        if player_index >= len(next_state.players) or player_index >= len(prev_state.players):
            return 0.0

        prev_player = prev_state.players[player_index]
        next_player = next_state.players[player_index]

        # 1. Step route points delta
        delta_score = 0.0
        is_claim = (action is not None and hasattr(action, "action_type") and action.action_type == ActionType.CLAIM_ROUTE)
        if is_claim:
            route = self._routes_by_id.get(action.route_id or "")
            if route:
                delta_score = float(GameRules.points_for_route_length(route.length))
        elif prev_player is not next_player:
            delta_score = float(next_player.score - prev_player.score)

        reward = self.weights.route_points_weight * delta_score - self.weights.step_penalty

        # 2. Ticket completion delta during the step (only relevant if route was claimed or tickets changed)
        if is_claim or len(next_player.claimed_route_ids) != len(prev_player.claimed_route_ids):
            prev_routes = [
                self._routes_by_id[rid]
                for rid in prev_player.claimed_route_ids
                if rid in self._routes_by_id
            ]
            next_routes = [
                self._routes_by_id[rid]
                for rid in next_player.claimed_route_ids
                if rid in self._routes_by_id
            ]

            prev_completed = check_tickets_completed_batch(prev_routes, prev_player.tickets)
            next_completed = check_tickets_completed_batch(next_routes, next_player.tickets)

            for t in next_player.tickets:
                was_done = prev_completed.get(t.id, False)
                is_done = next_completed.get(t.id, False)
                if not was_done and is_done:
                    reward += self.weights.ticket_completion_weight * float(t.points)

        # 3. Terminal outcome reward
        if next_state.is_game_over:
            opp_index = 1 - player_index if len(next_state.players) == 2 else None
            opp_score = next_state.players[opp_index].score if opp_index is not None else 0

            # Win/Loss outcome
            if next_state.winner_id == next_player.id:
                reward += self.weights.win_bonus
            elif next_state.winner_id is not None and next_state.winner_id != next_player.id:
                reward -= self.weights.loss_penalty

            # Score differential
            score_diff = float(next_player.score - opp_score)
            reward += self.weights.score_diff_weight * score_diff

            # Uncompleted tickets penalty
            terminal_routes = [
                self._routes_by_id[rid]
                for rid in next_player.claimed_route_ids
                if rid in self._routes_by_id
            ]
            final_completed = check_tickets_completed_batch(terminal_routes, next_player.tickets)
            for t in next_player.tickets:
                if not final_completed.get(t.id, False):
                    reward -= self.weights.ticket_failure_penalty_weight * float(t.points)

        return float(reward)
