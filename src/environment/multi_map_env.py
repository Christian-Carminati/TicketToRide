"""Gymnasium environment supporting multi-map training across procedural topologies."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import ObservationV1
from src.environment.reward import BaseRewardCalculator, RewardFactory
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.game import Game
from src.game.state import TurnState
from src.game.ticket import DestinationTicket


class MultiMapTicketToRideEnv(gym.Env):
    """Gymnasium environment that samples new map topologies from a dataset upon reset."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        maps: list[tuple[Board, list[DestinationTicket]]],
        sampling: Literal["random", "round_robin"] = "random",
        opponent: Any | None = None,
        reward_calculator: BaseRewardCalculator | str | int | None = None,
        num_players: int = 2,
        max_turns: int = 300,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not maps:
            raise ValueError("Maps list must not be empty.")

        self.maps = maps
        self.sampling = sampling
        self.num_players = num_players
        self.max_turns = max_turns
        self.rng = np.random.default_rng(seed)
        self._current_map_idx = 0

        if opponent is None:
            from src.agents.random_agent import RandomAgent

            self.opponent = RandomAgent(seed=seed)
        else:
            self.opponent = opponent

        self.reward_calculator_config = reward_calculator

        # Reference template for fixed action and observation space
        ref_board, ref_tickets = self.maps[0]
        self.ref_actions = DiscreteActionSpace(board=ref_board)
        self.ref_encoder = ObservationV1(
            board=ref_board, initial_tickets=ref_tickets, num_players=self.num_players
        )

        self.action_space: spaces.Discrete = spaces.Discrete(self.ref_actions.n)
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.ref_encoder.observation_shape,
            dtype=np.float32,
        )

        # Initialize with first map
        self._set_active_map(0, seed=seed)

    @property
    def current_board(self) -> Board:
        return self.game.board

    def _set_active_map(self, map_idx: int, seed: int | None = None) -> None:
        self._current_map_idx = map_idx
        board, tickets = self.maps[map_idx]

        self.game = Game(
            board=board,
            tickets_deck=tickets,
            num_players=self.num_players,
            seed=seed if seed is not None else 42,
        )
        self.encoder = ObservationV1(
            board=board,
            initial_tickets=tickets,
            num_players=self.num_players,
        )
        self.reward_calc = RewardFactory.create(self.reward_calculator_config, board=board)
        self.discrete_actions = DiscreteActionSpace(board=board)
        self.masker = ActionMasker(self.discrete_actions)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Select map
        if self.sampling == "random":
            chosen_idx = int(self.rng.integers(0, len(self.maps)))
        else:
            chosen_idx = self._current_map_idx % len(self.maps)
            self._current_map_idx += 1

        self._set_active_map(chosen_idx, seed=seed)
        self.game.reset(seed=seed)
        if self.opponent and hasattr(self.opponent, "reset"):
            self.opponent.reset(seed=seed)

        self._auto_step_opponents_if_needed()

        raw_obs = self.encoder.encode(self.game.state, player_index=0)
        obs = self._format_observation(raw_obs)
        valid_actions = self.game.valid_actions()
        pending = self.game.state.players[0].pending_tickets
        mask = self._format_action_mask(
            self.masker.compute_mask(valid_actions, pending_tickets=pending)
        )

        info = {
            "action_mask": mask,
            "turn": self.game.state.turn_number,
            "map_index": chosen_idx,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Map canonical action ID to active board discrete action
        active_action_id = min(action, self.discrete_actions.n - 1)
        domain_action = self._resolve_action(active_action_id)
        prev_state = self.game.state
        self.game.step(domain_action)

        self._auto_step_opponents_if_needed()

        if self.game.state.turn_number >= self.max_turns and not self.game.state.is_game_over:
            self.game._end_game()

        reward = self.reward_calc.calculate(
            prev_state=prev_state,
            action=domain_action,
            next_state=self.game.state,
            player_index=0,
        )

        terminated = bool(self.game.state.is_game_over)
        truncated = bool(self.game.state.turn_number >= self.max_turns and not terminated)

        raw_obs = self.encoder.encode(self.game.state, player_index=0)
        obs = self._format_observation(raw_obs)
        valid_actions = self.game.valid_actions() if not terminated else []
        pending = self.game.state.players[0].pending_tickets if not terminated else []
        mask = self._format_action_mask(
            self.masker.compute_mask(valid_actions, pending_tickets=pending)
        )

        info = {
            "action_mask": mask,
            "turn": self.game.state.turn_number,
            "winner_id": self.game.state.winner_id,
            "player_score": self.game.state.players[0].score,
        }

        return obs, float(reward), terminated, truncated, info

    def _format_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        target_dim = self.observation_space.shape[0]
        if len(raw_obs) == target_dim:
            return raw_obs
        formatted = np.zeros(target_dim, dtype=np.float32)
        copy_len = min(len(raw_obs), target_dim)
        formatted[:copy_len] = raw_obs[:copy_len]
        return formatted

    def _format_action_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        target_dim = int(self.action_space.n)
        if len(raw_mask) == target_dim:
            return raw_mask
        formatted = np.zeros(target_dim, dtype=bool)
        copy_len = min(len(raw_mask), target_dim)
        formatted[:copy_len] = raw_mask[:copy_len]
        return formatted

    def _resolve_action(self, action_id: int) -> Action:
        action = self.discrete_actions.to_action(action_id)

        if action.action_type == ActionType.KEEP_TICKETS:
            curr_player_idx = self.game.state.current_player_index
            pending = self.game.state.players[curr_player_idx].pending_tickets
            if action.ticket_ids and pending:
                actual_ticket_ids = tuple(
                    pending[int(idx)].id for idx in action.ticket_ids if int(idx) < len(pending)
                )
                return Action(
                    action_type=ActionType.KEEP_TICKETS,
                    ticket_ids=actual_ticket_ids,
                )

        if action.action_type == ActionType.CLAIM_ROUTE:
            route = self.game.board.get_route(action.route_id or "")
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
        if not self.opponent:
            return

        while (
            not self.game.state.is_game_over
            and self.game.state.current_player_index != 0
            and self.game.state.turn_number < self.max_turns
        ):
            valid_actions = self.game.valid_actions()
            if not valid_actions:
                if self.game.state.turn_state == TurnState.NORMAL:
                    self.game._end_game()
                break
            opp_action = self.opponent.act(
                self.game.state,
                valid_actions,
                self.game.board,
            )
            self.game.step(opp_action)
