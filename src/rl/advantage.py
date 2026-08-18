"""Generalized Advantage Estimation (GAE) for Actor-Critic methods."""

from collections.abc import Sequence

import numpy as np


def compute_gae(
    rewards: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    dones: Sequence[bool] | np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and Discounted Returns.

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
    Returns_t = A_t + V(s_t)
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        v_next = next_value if t == n_steps - 1 else float(values[t + 1])
        non_terminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * v_next * non_terminal - float(values[t])
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns
