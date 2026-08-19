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
    r_arr = np.asarray(rewards, dtype=np.float32)
    v_arr = np.asarray(values, dtype=np.float32)
    d_arr = np.asarray(dones, dtype=np.float32)
    n_steps = len(r_arr)

    if n_steps == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0
    gamma_lambda = gamma * gae_lambda

    for t in reversed(range(n_steps)):
        v_next = next_value if t == n_steps - 1 else v_arr[t + 1]
        non_terminal = 1.0 - d_arr[t]
        delta = r_arr[t] + gamma * v_next * non_terminal - v_arr[t]
        last_gae = delta + gamma_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + v_arr
    return advantages, returns
