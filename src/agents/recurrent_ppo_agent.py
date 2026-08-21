"""Recurrent PPO Agent for playing Ticket to Ride with internal LSTM memory."""

from typing import Any
import numpy as np
import torch

from src.agents.base_agent import BaseAgent
from src.environment.action_mask import compute_action_mask
from src.environment.action_space import ActionSpaceV1
from src.environment.observation import ObservationV1
from src.game.action import Action
from src.game.board import Board
from src.game.maps import load_usa_board
from src.game.state import GameState
from src.game.ticket import DestinationTicket
from src.rl.lstm_ppo import RecurrentMaskedActorCritic


class RecurrentPPOAgent(BaseAgent):
    """Agent driven by a Recurrent Actor-Critic policy with sequential memory."""

    def __init__(
        self,
        model: RecurrentMaskedActorCritic | str | None = None,
        board: Board | None = None,
        tickets: list[DestinationTicket] | None = None,
        num_players: int = 2,
        deterministic: bool = True,
        device: str = "cpu",
        name: str = "RecurrentPPOAgent",
        model_or_path: RecurrentMaskedActorCritic | str | None = None,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
    ) -> None:
        super().__init__(name=name)
        if board is None:
            self.board, self.tickets = load_usa_board()
        else:
            self.board = board
            self.tickets = tickets or []

        self.num_players = num_players
        self.deterministic = deterministic
        self.device = torch.device(device)

        self.obs_encoder = ObservationV1(
            board=self.board, initial_tickets=self.tickets, num_players=num_players
        )
        self.action_space = ActionSpaceV1(self.board)
        obs_dim = self.obs_encoder.observation_shape[0]
        action_dim = self.action_space.n

        target_model = model if model is not None else model_or_path

        if isinstance(target_model, str):
            checkpoint = torch.load(target_model, map_location=self.device)
            state_dict = None
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "actor_critic_state_dict" in checkpoint:
                    state_dict = checkpoint["actor_critic_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            c_hidden = checkpoint.get("hidden_dim", hidden_dim) if isinstance(checkpoint, dict) else hidden_dim
            c_lstm = (
                checkpoint.get("lstm_hidden_dim", lstm_hidden_dim)
                if isinstance(checkpoint, dict)
                else lstm_hidden_dim
            )
            if state_dict and "encoder.0.weight" in state_dict:
                c_hidden = state_dict["encoder.0.weight"].shape[0]
            if state_dict and "critic.weight" in state_dict:
                c_lstm = state_dict["critic.weight"].shape[1]

            self.model = RecurrentMaskedActorCritic(
                input_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=c_hidden,
                lstm_hidden_dim=c_lstm,
            ).to(self.device)
            self.model.load_state_dict(state_dict)
        elif isinstance(target_model, RecurrentMaskedActorCritic):
            self.model = target_model.to(self.device)
        else:
            self.model = RecurrentMaskedActorCritic(
                input_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
            ).to(self.device)

        self.model.eval()
        self.current_hidden = self.model.get_initial_hidden(batch_size=1, device=self.device)

    def reset(self, seed: int | None = None) -> None:
        """Reset internal recurrent hidden state for a new game."""
        super().reset(seed=seed)
        self.current_hidden = self.model.get_initial_hidden(batch_size=1, device=self.device)

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray | None = None,
        deterministic: bool | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Select discrete action index given observation vector and action mask."""
        det = self.deterministic if deterministic is None else deterministic
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = (
            torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            if action_mask is not None
            else None
        )

        with torch.no_grad():
            action_tensor, _, _, _, next_hidden = self.model.get_action_and_value(
                obs_tensor,
                self.current_hidden,
                action_mask=mask_tensor,
                deterministic=det,
            )
            self.current_hidden = next_hidden

        return int(action_tensor.item())

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        """Encode state, apply mask, run recurrent inference, and decode action."""
        if not valid_actions:
            raise ValueError("No valid actions available.")

        obs = self.obs_encoder.encode(state, player_index=state.current_player_index)
        pending = state.current_player.pending_tickets if state.current_player else None
        mask = compute_action_mask(valid_actions, self.action_space, pending_tickets=pending)

        action_idx = self.select_action(obs, mask)
        decoded = self.action_space.decode(action_idx, state)

        # Fallback to first valid action if decoding produces an invalid choice
        if decoded not in valid_actions:
            return valid_actions[0]
        return decoded

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hidden_dim": self.model.hidden_dim,
                "lstm_hidden_dim": self.model.lstm_hidden_dim,
                "input_dim": self.model.input_dim,
                "action_dim": self.model.action_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "actor_critic_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["actor_critic_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()


__all__ = ["RecurrentPPOAgent"]
