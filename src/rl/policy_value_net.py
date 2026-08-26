"""
Dual-headed Policy-Value Neural Network for AlphaZero-style MCTS.

Calculates:
- Policy Prior: P_theta(s, a) via masked logits Softmax
- State Value: v_theta(s) in [-1.0, 1.0] via Tanh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """Residual block with Linear, LayerNorm, and ReLU activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.relu(out + residual)


class PolicyValueNetwork(nn.Module):
    """
    Dual-Headed Actor-Critic Network for AlphaZero Search & Training.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        # Shared trunk
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action probabilities and scalar state value.
        """
        h = self.input_layer(obs)
        for block in self.res_blocks:
            h = block(h)

        logits = self.policy_head(h)
        if action_mask is not None:
            # Apply large negative penalty to masked invalid actions
            logits = torch.where(
                action_mask > 0.5,
                logits,
                -1e8,
            )

        policy_probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)
        return policy_probs, value

    def sync_fast_evaluator(self) -> None:
        """Synchronizes internal NumPy weights for zero-overhead MCTS evaluation."""
        if not hasattr(self, "_fast_evaluator") or self._fast_evaluator is None:
            self._fast_evaluator = FastNumpyEvaluator(self)
        else:
            self._fast_evaluator.sync()

    def evaluate_state(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Numpy inference wrapper for ultra-fast MCTS node evaluation.
        """
        if not hasattr(self, "_fast_evaluator") or self._fast_evaluator is None:
            self._fast_evaluator = FastNumpyEvaluator(self)
        return self._fast_evaluator.evaluate(obs, action_mask)


    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_res_blocks": self.num_res_blocks,
                "state_dict": self.state_dict(),
            },
            p,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> PolicyValueNetwork:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        net = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_res_blocks=checkpoint.get("num_res_blocks", 2),
        )
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        return net


class FastNumpyEvaluator:
    """Zero-overhead NumPy inference evaluator for scalar MCTS leaf node evaluation."""

    def __init__(self, net: PolicyValueNetwork):
        self.net = net
        self.weights: dict[str, np.ndarray] = {}
        self.sync()

    def sync(self) -> None:
        self.weights = {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def evaluate(
        self, obs: np.ndarray, action_mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        w = self.weights
        # Trunk input
        h = obs @ w["input_layer.0.weight"].T + w["input_layer.0.bias"]
        mean = np.mean(h, axis=-1, keepdims=True)
        var = np.var(h, axis=-1, keepdims=True)
        h = (h - mean) / np.sqrt(var + 1e-5) * w["input_layer.1.weight"] + w["input_layer.1.bias"]
        h = np.maximum(0.0, h)

        # Res blocks
        for i in range(self.net.num_res_blocks):
            residual = h
            pfx = f"res_blocks.{i}."
            h1 = h @ w[f"{pfx}fc1.weight"].T + w[f"{pfx}fc1.bias"]
            m1 = np.mean(h1, axis=-1, keepdims=True)
            v1 = np.var(h1, axis=-1, keepdims=True)
            h1 = (h1 - m1) / np.sqrt(v1 + 1e-5) * w[f"{pfx}ln1.weight"] + w[f"{pfx}ln1.bias"]
            h1 = np.maximum(0.0, h1)

            h2 = h1 @ w[f"{pfx}fc2.weight"].T + w[f"{pfx}fc2.bias"]
            m2 = np.mean(h2, axis=-1, keepdims=True)
            v2 = np.var(h2, axis=-1, keepdims=True)
            h2 = (h2 - m2) / np.sqrt(v2 + 1e-5) * w[f"{pfx}ln2.weight"] + w[f"{pfx}ln2.bias"]
            h = np.maximum(0.0, h2 + residual)

        # Policy head
        hp = np.maximum(0.0, h @ w["policy_head.0.weight"].T + w["policy_head.0.bias"])
        logits = hp @ w["policy_head.2.weight"].T + w["policy_head.2.bias"]
        if action_mask is not None:
            logits = np.where(action_mask > 0.5, logits, -1e8)
        max_logit = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits / np.maximum(sum_exp, 1e-12)

        # Value head
        hv = np.maximum(0.0, h @ w["value_head.0.weight"].T + w["value_head.0.bias"])
        v_logit = hv @ w["value_head.2.weight"].T + w["value_head.2.bias"]
        val = float(np.tanh(v_logit).item())
        return probs.astype(np.float32), val
