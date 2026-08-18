"""Gymnasium Environment wrapper for Ticket to Ride."""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.environment.reward import BaseRewardCalculator, DefaultRewardCalculator
from src.game.game import Game


class TicketToRideEnv(gym.Env):
    """Gymnasium-compatible single-agent / turn-based environment."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        game: Game | None = None,
        observation_encoder: BaseObservationEncoder | None = None,
        reward_calculator: BaseRewardCalculator | None = None,
    ) -> None:
        super().__init__()
        self.game = game or Game()
        self.encoder = observation_encoder or ObservationV1()
        self.reward_calc = reward_calculator or DefaultRewardCalculator()
        self.discrete_actions = DiscreteActionSpace()
        self.masker = ActionMasker(self.discrete_actions)

        self.action_space = spaces.Discrete(self.discrete_actions.n)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
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
        obs = self.encoder.encode(state, player_index=state.current_player_index)
        valid_actions = self.game.valid_actions()
        action_mask = self.masker.compute_mask(valid_actions)
        info = {"action_mask": action_mask, "turn": state.turn_number}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        domain_action = self.discrete_actions.to_action(action)
        prev_state = self.game.state
        next_state = self.game.step(domain_action)

        reward = self.reward_calc.calculate(
            prev_state=prev_state,
            action=domain_action,
            next_state=next_state,
            player_index=prev_state.current_player_index,
        )

        terminated = next_state.is_game_over
        truncated = False
        obs = self.encoder.encode(next_state, player_index=next_state.current_player_index)
        valid_actions = self.game.valid_actions()
        action_mask = self.masker.compute_mask(valid_actions)
        info = {"action_mask": action_mask, "winner_id": next_state.winner_id}

        return obs, reward, terminated, truncated, info
