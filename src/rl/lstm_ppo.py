"""Recurrent Actor-Critic network architecture and PPO trainer for POMDPs."""

import math
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize linear layers with orthogonal weights and constant bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RecurrentMaskedActorCritic(nn.Module):
    """Recurrent Actor-Critic architecture with LSTM memory and action masking."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
        orthogonal_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        if orthogonal_init:
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden_dim), gain=np.sqrt(2)),
                nn.Tanh(),
            )
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            for name, param in self.lstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            self.actor = layer_init(nn.Linear(lstm_hidden_dim, action_dim), gain=0.01)
            self.critic = layer_init(nn.Linear(lstm_hidden_dim, 1), gain=1.0)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
            )
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            self.actor = nn.Linear(lstm_hidden_dim, action_dim)
            self.critic = nn.Linear(lstm_hidden_dim, 1)

    def get_initial_hidden(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero initial hidden states (h, c) for LSTM."""
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, dtype=torch.float32, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, dtype=torch.float32, device=device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass across either single step or sequence tensor."""
        if x.dim() == 2:
            # (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        features = self.encoder(x)
        lstm_out, new_hidden = self.lstm(features, hidden)
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out)

        if squeeze_output:
            return logits.squeeze(1), values.squeeze(1), new_hidden
        return logits, values, new_hidden

    def get_action_and_value(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute action, log_prob, entropy, and value with optional action masking."""
        logits, values, new_hidden = self.forward(x, hidden)

        if action_mask is not None:
            if action_mask.dim() == 2 and logits.dim() == 3:
                action_mask = action_mask.unsqueeze(1)
            masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))
        else:
            masked_logits = logits

        dist = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, values, new_hidden

    def get_value(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute state value and return updated hidden state."""
        _, values, new_hidden = self.forward(x, hidden)
        return values, new_hidden


# Alias for backward compatibility
RecurrentPPOActorCritic = RecurrentMaskedActorCritic

# Re-export RecurrentRolloutBuffer
from src.rl.rollout import RecurrentRolloutBuffer  # noqa: E402

__all__ = [
    "RecurrentMaskedActorCritic",
    "RecurrentPPOActorCritic",
    "RecurrentRolloutBuffer",
    "layer_init",
]


