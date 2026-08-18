"""Generalized Advantage Estimation (GAE)."""


import numpy as np


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation and Returns.

    A_t = delta_t + (gamma * lambda) * (1 - done_{t+1}) * A_{t+1}
    where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        v_next = next_value if t == n_steps - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * v_next * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns
