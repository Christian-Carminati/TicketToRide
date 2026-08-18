"""PyTorch Neural Network architectures for DQN and PPO with Action Masking."""

import random

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


class MaskedQNetwork(nn.Module):
    """Deep Q-Network with explicit Action Masking and epsilon-greedy exploration."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unmasked Q-values for all actions."""
        return self.net(x)

    def select_action(
        self,
        obs: torch.Tensor,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        """Select an action using epsilon-greedy strategy over legally valid actions."""
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            return 0

        if random.random() < epsilon:
            return int(random.choice(valid_indices))

        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            q_values = self.forward(obs).squeeze(0).clone()
            # Mask invalid actions with large negative value (-1e9)
            mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)
            q_values[~mask_tensor] = -1e9
            return int(torch.argmax(q_values).item())


class MaskedActorCritic(nn.Module):
    """Discrete Actor-Critic architecture with Action Masking for PPO."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action sampling, log-probabilities, entropy, and state value."""
        logits = self.actor(obs)
        value = self.critic(obs)

        if action_mask is not None:
            # Mask out invalid actions with -1e8 before softmax
            masked_logits = torch.where(
                action_mask,
                logits,
                torch.tensor(-1e8, dtype=logits.dtype, device=logits.device),
            )
        else:
            masked_logits = logits

        dist = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


# Backwards-compatibility aliases for Phase 1-3 tests
QNetworkMLP = MaskedQNetwork
ActorCriticMLP = MaskedActorCritic
