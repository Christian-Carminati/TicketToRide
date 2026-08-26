"""PPO Agent for inference and evaluation."""

from typing import Any

import numpy as np
import torch

from src.agents.base_agent import BaseAgent
from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.state import GameState
from src.game.ticket import DestinationTicket
from src.rl.networks import MaskedActorCritic


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent for tournament and environment play."""

    def __init__(
        self,
        name: str = "PPOAgent",
        model_path: str | None = None,
        actor_critic: MaskedActorCritic | None = None,
        model: MaskedActorCritic | None = None,
        board: Board | None = None,
        tickets: list[DestinationTicket] | None = None,
        input_dim: int = 100,
        action_dim: int = 56,
        hidden_dim: int = 128,
        device: str = "cpu",
        encoder: BaseObservationEncoder | None = None,
        discrete_actions: DiscreteActionSpace | None = None,
    ) -> None:
        super().__init__(name=name)
        self.device = device
        ac = actor_critic if actor_critic is not None else model
        if board is not None:
            self.board = board
            self.tickets = tickets or []
            if encoder is None:
                encoder = ObservationV1(board=board, initial_tickets=self.tickets, num_players=2)
            if discrete_actions is None:
                discrete_actions = DiscreteActionSpace(board=board)
        self.encoder = encoder
        self.discrete_actions = discrete_actions
        self.masker = (
            ActionMasker(self.discrete_actions) if self.discrete_actions is not None else None
        )
        if ac is not None:
            self.actor_critic = ac.to(device)
        else:
            self.actor_critic = MaskedActorCritic(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            ).to(device)
        self.actor_critic.eval()
        if model_path is not None:
            self.load(model_path)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        deterministic: bool = True,
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
                obs_tensor, action_mask=mask_tensor, deterministic=deterministic
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

        if (
            self.encoder is not None
            and self.discrete_actions is not None
            and self.masker is not None
        ):
            obs = self.encoder.encode(state, player_index=state.current_player_index)
            pending = state.current_player.pending_tickets if state.current_player else None
            mask = self.masker.compute_mask(valid_actions, pending_tickets=pending)

            chosen_idx = self.select_action(obs, action_mask=mask)
            domain_act = self.discrete_actions.to_action(chosen_idx)

            if (
                domain_act.action_type == ActionType.KEEP_TICKETS
                and state.current_player
                and pending
            ):
                slots = {int(idx) for idx in (domain_act.ticket_ids or ())}
                chosen_ids = tuple(t.id for slot_i, t in enumerate(pending) if slot_i in slots)
                for va in valid_actions:
                    if va.action_type == ActionType.KEEP_TICKETS and set(
                        va.ticket_ids or ()
                    ) == set(chosen_ids):
                        return va

            for va in valid_actions:
                if self.discrete_actions.to_id(va) == chosen_idx:
                    return va

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
