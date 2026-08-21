"""Gymnasium Environment wrapper for Ticket to Ride."""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.environment.reward import BaseRewardCalculator, RewardFactory
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.game import Game
from src.game.maps import load_usa_board
from src.game.state import TurnState
from src.game.ticket import DestinationTicket


class TicketToRideEnv(gym.Env):
    """Gymnasium-compatible single-agent / turn-based environment with auto-stepping opponent."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        board: Board | None = None,
        tickets_deck: list[DestinationTicket] | None = None,
        opponent: Any | None = None,
        observation_encoder: BaseObservationEncoder | None = None,
        reward_calculator: BaseRewardCalculator | str | int | None = None,
        num_players: int = 2,
        max_turns: int = 300,
        board_type: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if board is not None:
            self.board = board
            self.initial_tickets = tickets_deck or []
        elif board_type == "mini":
            from src.game.maps import create_synthetic_mini_board

            self.board, self.initial_tickets = create_synthetic_mini_board()
        else:
            self.board, self.initial_tickets = load_usa_board()

        self.num_players = num_players
        self.max_turns = max_turns
        if opponent is None:
            from src.agents.random_agent import RandomAgent

            self.opponent = RandomAgent(seed=seed if seed is not None else 42)
        else:
            self.opponent = opponent

        self.game = Game(
            board=self.board,
            tickets_deck=self.initial_tickets,
            num_players=self.num_players,
            seed=seed if seed is not None else 42,
        )
        self.encoder = observation_encoder or ObservationV1(
            board=self.board,
            initial_tickets=self.initial_tickets,
            num_players=self.num_players,
        )
        self.reward_calc = RewardFactory.create(reward_calculator, board=self.board)
        self.discrete_actions = DiscreteActionSpace(board=self.board)
        self.masker = ActionMasker(self.discrete_actions)

        self.action_space = spaces.Discrete(self.discrete_actions.n)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.encoder.observation_shape,
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        state = self.game.reset(seed=seed)
        if self.opponent and hasattr(self.opponent, "reset"):
            self.opponent.reset(seed=seed)

        # Handle initial tickets selection if opponent is configured
        self._auto_step_opponents_if_needed()

        obs = self.encoder.encode(self.game.state, player_index=0)
        valid_actions = self.game.valid_actions()
        pending = self.game.state.players[0].pending_tickets
        action_mask = self.masker.compute_mask(valid_actions, pending_tickets=pending)

        info = {
            "action_mask": action_mask,
            "turn": self.game.state.turn_number,
            "turn_state": self.game.state.turn_state.value,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        domain_action = self._resolve_action(action)
        prev_state = self.game.state
        self.game.step(domain_action)

        # Auto-step opponent(s) until it is Player 0's turn again or game is over
        self._auto_step_opponents_if_needed()

        # Check turn limit truncation
        if self.game.state.turn_number >= self.max_turns and not self.game.state.is_game_over:
            self.game._end_game()

        # Calculate reward and component breakdown from the perspective of Player 0
        reward = self.reward_calc.calculate(
            prev_state=prev_state,
            action=domain_action,
            next_state=self.game.state,
            player_index=0,
        )
        reward_components = self.reward_calc.get_components(
            prev_state=prev_state,
            action=domain_action,
            next_state=self.game.state,
            player_index=0,
        )

        terminated = bool(self.game.state.is_game_over)
        truncated = bool(self.game.state.turn_number >= self.max_turns and not terminated)

        obs = self.encoder.encode(self.game.state, player_index=0)
        valid_actions = self.game.valid_actions() if not terminated else []
        pending = self.game.state.players[0].pending_tickets if not terminated else []
        action_mask = self.masker.compute_mask(valid_actions, pending_tickets=pending)

        info = {
            "action_mask": action_mask,
            "winner_id": self.game.state.winner_id,
            "turn": self.game.state.turn_number,
            "player_score": self.game.state.players[0].score,
            "opponent_score": (
                self.game.state.players[1].score if len(self.game.state.players) > 1 else 0
            ),
            "reward_components": reward_components,
        }

        return obs, float(reward), terminated, truncated, info

    def _resolve_action(self, action_id: int) -> Action:
        """Map discrete action index to concrete domain Action, resolving tickets and locomotive counts."""
        action = self.discrete_actions.to_action(action_id)

        if action.action_type == ActionType.KEEP_TICKETS:
            # Map subset index strings ('0', '1', ...) to actual ticket IDs in pending_tickets
            curr_player_idx = self.game.state.current_player_index
            pending = self.game.state.players[curr_player_idx].pending_tickets
            if action.ticket_ids and pending:
                actual_ticket_ids = tuple(
                    pending[int(idx)].id
                    for idx in action.ticket_ids
                    if int(idx) < len(pending)
                )
                return Action(
                    action_type=ActionType.KEEP_TICKETS,
                    ticket_ids=actual_ticket_ids,
                )

        if action.action_type == ActionType.CLAIM_ROUTE:
            route = self.board.get_route(action.route_id or "")
            if route and action.color_chosen:
                curr_player_idx = self.game.state.current_player_index
                player = self.game.state.players[curr_player_idx]
                color_count = player.cards.get(action.color_chosen, 0)
                needed_locos = max(0, route.length - color_count)
                return Action(
                    action_type=ActionType.CLAIM_ROUTE,
                    route_id=route.id,
                    color_chosen=action.color_chosen,
                    locomotives_count=needed_locos,
                )

        return action

    def _auto_step_opponents_if_needed(self) -> None:
        """Step configured opponent(s) until player 0's turn or game over."""
        if not self.opponent:
            return

        while (
            not self.game.state.is_game_over
            and self.game.state.current_player_index != 0
            and self.game.state.turn_number < self.max_turns
        ):
            valid_actions = self.game.valid_actions()
            if not valid_actions:
                # If player in NORMAL state has no valid actions, end game cleanly
                if self.game.state.turn_state == TurnState.NORMAL:
                    self.game._end_game()
                break
            opp_action = self.opponent.act(
                self.game.state,
                valid_actions,
                self.game.board,
            )
            self.game.step(opp_action)
