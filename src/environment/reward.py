"""Configurable, modular, and versioned reward calculation engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from src.game.action import ActionType
from src.game.board import Board
from src.game.graph import check_tickets_completed_batch
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
    """Abstract interface for modular reward calculation engines."""

    def __init__(
        self,
        weights: RewardWeights | None = None,
        board: Board | None = None,
    ) -> None:
        pass

    @abstractmethod
    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        """Compute the scalar step reward for player_index."""

    @abstractmethod
    def get_components(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> dict[str, float]:
        """Compute the detailed breakdown of reward components for telemetry and introspection."""


class RewardV1_Sparse(BaseRewardCalculator):
    """Reward V1: Pure terminal outcome reward with zero intermediate step shaping.

    Focuses on studying the credit assignment problem on long episodic horizons.
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        board: Board | None = None,
    ) -> None:
        self.weights = weights or RewardWeights(
            win_bonus=20.0,
            loss_penalty=10.0,
            score_diff_weight=0.5,
            ticket_failure_penalty_weight=1.0,
        )
        if board is None:
            usa_board, _ = load_usa_board()
            self._routes_by_id = {r.id: r for r in usa_board.routes}
        else:
            self._routes_by_id = {r.id: r for r in board.routes}

    def get_components(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> dict[str, float]:
        components: dict[str, float] = {
            "step_reward": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "score_diff": 0.0,
            "uncompleted_tickets_penalty": 0.0,
        }
        if player_index >= len(next_state.players):
            return components

        if next_state.is_game_over:
            player = next_state.players[player_index]
            opp_index = 1 - player_index if len(next_state.players) == 2 else None
            opp_score = next_state.players[opp_index].score if opp_index is not None else 0

            # Win/Loss outcome
            if next_state.winner_id == player.id:
                components["win_bonus"] = float(self.weights.win_bonus)
            elif next_state.winner_id is not None and next_state.winner_id != player.id:
                components["loss_penalty"] = float(-self.weights.loss_penalty)

            # Score differential
            components["score_diff"] = float(
                self.weights.score_diff_weight * (player.score - opp_score)
            )

            # Uncompleted tickets penalty
            terminal_routes = [
                self._routes_by_id[rid]
                for rid in player.claimed_route_ids
                if rid in self._routes_by_id
            ]
            final_completed = check_tickets_completed_batch(terminal_routes, player.tickets)
            ticket_pen = sum(
                float(t.points) for t in player.tickets if not final_completed.get(t.id, False)
            )
            components["uncompleted_tickets_penalty"] = float(
                -self.weights.ticket_failure_penalty_weight * ticket_pen
            )

        return components

    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        components = self.get_components(prev_state, action, next_state, player_index)
        return float(sum(components.values()))


class RewardV2_DenseRoutes(BaseRewardCalculator):
    """Reward V2: Dense route points shaping with step penalty and terminal outcomes.

    Incentivizes immediate claiming of high-value routes and rapid game progression.
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        board: Board | None = None,
    ) -> None:
        self.weights = weights or RewardWeights(
            route_points_weight=1.0,
            step_penalty=0.01,
            win_bonus=20.0,
            loss_penalty=10.0,
            score_diff_weight=0.5,
            ticket_failure_penalty_weight=1.0,
        )
        if board is None:
            usa_board, _ = load_usa_board()
            self._routes_by_id = {r.id: r for r in usa_board.routes}
        else:
            self._routes_by_id = {r.id: r for r in board.routes}

    def get_components(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> dict[str, float]:
        components: dict[str, float] = {
            "route_points": 0.0,
            "step_penalty": float(-self.weights.step_penalty),
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "score_diff": 0.0,
            "uncompleted_tickets_penalty": 0.0,
        }
        if player_index >= len(next_state.players) or player_index >= len(prev_state.players):
            return components

        prev_player = prev_state.players[player_index]
        next_player = next_state.players[player_index]

        # 1. Step route points delta
        delta_score = 0.0
        is_claim = (
            action is not None
            and hasattr(action, "action_type")
            and action.action_type == ActionType.CLAIM_ROUTE
        )
        if is_claim:
            route = self._routes_by_id.get(action.route_id or "")
            if route:
                delta_score = float(GameRules.points_for_route_length(route.length))
        elif prev_player is not next_player:
            delta_score = float(next_player.score - prev_player.score)

        components["route_points"] = float(self.weights.route_points_weight * delta_score)

        # 2. Terminal outcomes
        if next_state.is_game_over:
            opp_index = 1 - player_index if len(next_state.players) == 2 else None
            opp_score = next_state.players[opp_index].score if opp_index is not None else 0

            if next_state.winner_id == next_player.id:
                components["win_bonus"] = float(self.weights.win_bonus)
            elif next_state.winner_id is not None and next_state.winner_id != next_player.id:
                components["loss_penalty"] = float(-self.weights.loss_penalty)

            components["score_diff"] = float(
                self.weights.score_diff_weight * (next_player.score - opp_score)
            )

            terminal_routes = [
                self._routes_by_id[rid]
                for rid in next_player.claimed_route_ids
                if rid in self._routes_by_id
            ]
            final_completed = check_tickets_completed_batch(terminal_routes, next_player.tickets)
            ticket_pen = sum(
                float(t.points) for t in next_player.tickets if not final_completed.get(t.id, False)
            )
            components["uncompleted_tickets_penalty"] = float(
                -self.weights.ticket_failure_penalty_weight * ticket_pen
            )

        return components

    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        components = self.get_components(prev_state, action, next_state, player_index)
        return float(sum(components.values()))


class RewardV3_TicketMilestones(BaseRewardCalculator):
    """Reward V3: Ticket-focused shaping emphasizing real-time milestone completions and topological connectivity."""

    def __init__(
        self,
        weights: RewardWeights | None = None,
        board: Board | None = None,
    ) -> None:
        self.weights = weights or RewardWeights(
            route_points_weight=0.5,
            ticket_completion_weight=2.0,
            step_penalty=0.01,
            win_bonus=20.0,
            loss_penalty=10.0,
            score_diff_weight=0.5,
            ticket_failure_penalty_weight=2.0,
        )
        if board is None:
            usa_board, _ = load_usa_board()
            self._routes_by_id = {r.id: r for r in usa_board.routes}
        else:
            self._routes_by_id = {r.id: r for r in board.routes}

    def get_components(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> dict[str, float]:
        components: dict[str, float] = {
            "route_points": 0.0,
            "ticket_completion": 0.0,
            "step_penalty": float(-self.weights.step_penalty),
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "score_diff": 0.0,
            "uncompleted_tickets_penalty": 0.0,
        }
        if player_index >= len(next_state.players) or player_index >= len(prev_state.players):
            return components

        prev_player = prev_state.players[player_index]
        next_player = next_state.players[player_index]

        # 1. Route points delta
        delta_score = 0.0
        is_claim = (
            action is not None
            and hasattr(action, "action_type")
            and action.action_type == ActionType.CLAIM_ROUTE
        )
        if is_claim:
            route = self._routes_by_id.get(action.route_id or "")
            if route:
                delta_score = float(GameRules.points_for_route_length(route.length))
        elif prev_player is not next_player:
            delta_score = float(next_player.score - prev_player.score)

        components["route_points"] = float(self.weights.route_points_weight * delta_score)

        # 2. Ticket completion delta during the step
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

            ticket_points_awarded = 0.0
            for t in next_player.tickets:
                was_done = prev_completed.get(t.id, False)
                is_done = next_completed.get(t.id, False)
                if not was_done and is_done:
                    ticket_points_awarded += float(t.points)

            components["ticket_completion"] = float(
                self.weights.ticket_completion_weight * ticket_points_awarded
            )

        # 3. Terminal outcome reward
        if next_state.is_game_over:
            opp_index = 1 - player_index if len(next_state.players) == 2 else None
            opp_score = next_state.players[opp_index].score if opp_index is not None else 0

            if next_state.winner_id == next_player.id:
                components["win_bonus"] = float(self.weights.win_bonus)
            elif next_state.winner_id is not None and next_state.winner_id != next_player.id:
                components["loss_penalty"] = float(-self.weights.loss_penalty)

            components["score_diff"] = float(
                self.weights.score_diff_weight * (next_player.score - opp_score)
            )

            terminal_routes = [
                self._routes_by_id[rid]
                for rid in next_player.claimed_route_ids
                if rid in self._routes_by_id
            ]
            final_completed = check_tickets_completed_batch(terminal_routes, next_player.tickets)
            ticket_pen = sum(
                float(t.points) for t in next_player.tickets if not final_completed.get(t.id, False)
            )
            components["uncompleted_tickets_penalty"] = float(
                -self.weights.ticket_failure_penalty_weight * ticket_pen
            )

        return components

    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        components = self.get_components(prev_state, action, next_state, player_index)
        return float(sum(components.values()))


class RewardV4_StrategicShaped(BaseRewardCalculator):
    """Reward V4: Balanced strategic shaping uniting route points, ticket completion, efficiency, and score differential."""

    def __init__(
        self,
        weights: RewardWeights | None = None,
        board: Board | None = None,
    ) -> None:
        self.weights = weights or RewardWeights(
            route_points_weight=1.0,
            ticket_completion_weight=1.0,
            step_penalty=0.005,
            win_bonus=20.0,
            loss_penalty=10.0,
            score_diff_weight=0.5,
            ticket_failure_penalty_weight=1.0,
        )
        if board is None:
            usa_board, _ = load_usa_board()
            self._routes_by_id = {r.id: r for r in usa_board.routes}
        else:
            self._routes_by_id = {r.id: r for r in board.routes}

    def get_components(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> dict[str, float]:
        components: dict[str, float] = {
            "route_points": 0.0,
            "ticket_completion": 0.0,
            "step_penalty": float(-self.weights.step_penalty),
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "score_diff": 0.0,
            "uncompleted_tickets_penalty": 0.0,
        }
        if player_index >= len(next_state.players) or player_index >= len(prev_state.players):
            return components

        prev_player = prev_state.players[player_index]
        next_player = next_state.players[player_index]

        # 1. Route points delta
        delta_score = 0.0
        is_claim = (
            action is not None
            and hasattr(action, "action_type")
            and action.action_type == ActionType.CLAIM_ROUTE
        )
        if is_claim:
            route = self._routes_by_id.get(action.route_id or "")
            if route:
                delta_score = float(GameRules.points_for_route_length(route.length))
        elif prev_player is not next_player:
            delta_score = float(next_player.score - prev_player.score)

        components["route_points"] = float(self.weights.route_points_weight * delta_score)

        # 2. Ticket completion delta
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

            ticket_points_awarded = 0.0
            for t in next_player.tickets:
                was_done = prev_completed.get(t.id, False)
                is_done = next_completed.get(t.id, False)
                if not was_done and is_done:
                    ticket_points_awarded += float(t.points)

            components["ticket_completion"] = float(
                self.weights.ticket_completion_weight * ticket_points_awarded
            )

        # 3. Terminal outcomes
        if next_state.is_game_over:
            opp_index = 1 - player_index if len(next_state.players) == 2 else None
            opp_score = next_state.players[opp_index].score if opp_index is not None else 0

            if next_state.winner_id == next_player.id:
                components["win_bonus"] = float(self.weights.win_bonus)
            elif next_state.winner_id is not None and next_state.winner_id != next_player.id:
                components["loss_penalty"] = float(-self.weights.loss_penalty)

            components["score_diff"] = float(
                self.weights.score_diff_weight * (next_player.score - opp_score)
            )

            terminal_routes = [
                self._routes_by_id[rid]
                for rid in next_player.claimed_route_ids
                if rid in self._routes_by_id
            ]
            final_completed = check_tickets_completed_batch(terminal_routes, next_player.tickets)
            ticket_pen = sum(
                float(t.points) for t in next_player.tickets if not final_completed.get(t.id, False)
            )
            components["uncompleted_tickets_penalty"] = float(
                -self.weights.ticket_failure_penalty_weight * ticket_pen
            )

        return components

    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        components = self.get_components(prev_state, action, next_state, player_index)
        return float(sum(components.values()))


class CustomRewardCalculator(RewardV4_StrategicShaped):
    """Custom reward calculator initialized with arbitrary user-defined weights."""

    def __init__(
        self,
        weights: RewardWeights,
        board: Board | None = None,
    ) -> None:
        super().__init__(weights=weights, board=board)


# Backward-compatible alias for existing imports
DefaultRewardCalculator = RewardV4_StrategicShaped


class RewardFactory:
    """Factory for creating and registering reward calculators by version or alias."""

    _REGISTRY: ClassVar[dict[str, type[BaseRewardCalculator]]] = {
        "1": RewardV1_Sparse,
        "v1": RewardV1_Sparse,
        "sparse": RewardV1_Sparse,
        "outcome": RewardV1_Sparse,
        "2": RewardV2_DenseRoutes,
        "v2": RewardV2_DenseRoutes,
        "dense": RewardV2_DenseRoutes,
        "dense_routes": RewardV2_DenseRoutes,
        "routes": RewardV2_DenseRoutes,
        "3": RewardV3_TicketMilestones,
        "v3": RewardV3_TicketMilestones,
        "tickets": RewardV3_TicketMilestones,
        "ticket_milestones": RewardV3_TicketMilestones,
        "milestones": RewardV3_TicketMilestones,
        "4": RewardV4_StrategicShaped,
        "v4": RewardV4_StrategicShaped,
        "strategic": RewardV4_StrategicShaped,
        "balanced": RewardV4_StrategicShaped,
        "default": RewardV4_StrategicShaped,
        "custom": CustomRewardCalculator,
    }

    @classmethod
    def create(
        cls,
        version_or_name: int | str | BaseRewardCalculator | None,
        board: Board | None = None,
        weights: RewardWeights | None = None,
    ) -> BaseRewardCalculator:
        """Instantiate a reward calculator from a version number, alias string, or existing calculator."""
        if isinstance(version_or_name, BaseRewardCalculator):
            return version_or_name

        if version_or_name is None:
            key = "default"
        else:
            key = str(version_or_name).strip().lower()

        calc_class = cls._REGISTRY.get(key)
        if calc_class is None:
            raise ValueError(
                f"Unknown reward version or alias: '{version_or_name}'. "
                f"Available options: {list(cls._REGISTRY.keys())}"
            )

        if calc_class == CustomRewardCalculator:
            return CustomRewardCalculator(weights=weights or RewardWeights(), board=board)

        return calc_class(weights=weights, board=board)
