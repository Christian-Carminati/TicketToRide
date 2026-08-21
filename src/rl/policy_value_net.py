"""
Dual-headed Policy-Value Neural Network for AlphaZero-style MCTS.

Calculates:
- Policy Prior: P_theta(s, a) via masked logits Softmax
- State Value: v_theta(s) in [-1.0, 1.0] via Tanh
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
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
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action probabilities and scalar state value.
        """
        h = self.input_layer(obs)
        for block in self.res_blocks:
            h = block(h)
            
        logits = self.policy_head(h)
        if action_mask is not None:
            # Apply large negative penalty to masked invalid actions
            logits = torch.where(action_mask > 0.5, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))
            
        policy_probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)
        return policy_probs, value

    def evaluate_state(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Numpy inference wrapper for MCTS node evaluation.
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(action_mask, dtype=torch.float32).unsqueeze(0) if action_mask is not None else None
            p_t, v_t = self.forward(obs_t, mask_t)
            probs = p_t.squeeze(0).cpu().numpy()
            val = float(v_t.item())
        return probs, val

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_res_blocks": self.num_res_blocks,
            "state_dict": self.state_dict(),
        }, p)

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
