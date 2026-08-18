"""Recurrent Actor-Critic network architecture for Partially Observable MDPs."""


import torch
from torch import nn


class RecurrentPPOActorCritic(nn.Module):
    """LSTM-based Actor-Critic for handling partial observability (Phase 8)."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.actor = nn.Linear(lstm_hidden_dim, action_dim)
        self.critic = nn.Linear(lstm_hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features = self.encoder(x)
        lstm_out, new_hidden = self.lstm(features, hidden)
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return logits, value, new_hidden
