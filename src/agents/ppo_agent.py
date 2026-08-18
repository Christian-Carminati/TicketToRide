"""PPO Agent for inference and evaluation."""

from typing import Any

import numpy as np
import torch

from src.agents.base_agent import BaseAgent
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder
from src.game.action import Action
from src.game.board import Board
from src.game.state import GameState
from src.rl.networks import MaskedActorCritic


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent for tournament and environment play."""

    def __init__(
        self,
        name: str = "PPOAgent",
        model_path: str | None = None,
        input_dim: int = 100,
        action_dim: int = 56,
        hidden_dim: int = 128,
        device: str = "cpu",
        encoder: BaseObservationEncoder | None = None,
        discrete_actions: DiscreteActionSpace | None = None,
    ) -> None:
        super().__init__(name=name)
        self.device = device
        self.encoder = encoder
        self.discrete_actions = discrete_actions
        self.actor_critic = MaskedActorCritic(input_dim=input_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        self.actor_critic.eval()
        if model_path is not None:
            self.load(model_path)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = (
            torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            if action_mask is not None
            else None
        )

        with torch.no_grad():
            action, _, _, _ = self.actor_critic.get_action_and_value(
                obs_tensor, action_mask=mask_tensor, deterministic=True
            )
        return int(action.item())

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        if not valid_actions:
            raise ValueError("No valid actions available")

        if self.encoder is not None and self.discrete_actions is not None:
            obs = self.encoder.encode(state, player_id=state.current_player)
            mask = np.zeros(self.discrete_actions.size, dtype=np.int8)
            action_map = {}
            for action in valid_actions:
                idx = self.discrete_actions.encode_action(action)
                mask[idx] = 1
                action_map[idx] = action

            chosen_idx = self.select_action(obs, action_mask=mask)
            if chosen_idx in action_map:
                return action_map[chosen_idx]

        return valid_actions[0]

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "actor_critic_state_dict" in checkpoint:
            self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.actor_critic.load_state_dict(checkpoint["state_dict"])
        else:
            self.actor_critic.load_state_dict(checkpoint)
        self.actor_critic.eval()

    def save(self, path: str) -> None:
        torch.save({"actor_critic_state_dict": self.actor_critic.state_dict()}, path)
