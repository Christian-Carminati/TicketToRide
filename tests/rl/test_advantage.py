import numpy as np

from src.rl.advantage import compute_gae


def test_compute_gae_simple_trajectory() -> None:
    rewards = [1.0, 1.0, 2.0]
    values = [0.5, 0.8, 1.0]
    dones = [False, False, True]
    next_value = 0.0

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert len(advantages) == 3
    assert len(returns) == 3
    # Step 2: terminal step (done=True), delta = 2.0 + 0 - 1.0 = 1.0
    assert np.isclose(advantages[2], 1.0, atol=1e-4)
    assert np.isclose(returns[2], 2.0, atol=1e-4)


def test_compute_gae_intermediate_steps() -> None:
    # 2 non-terminal steps
    rewards = [1.0, 2.0]
    values = [1.0, 1.0]
    dones = [False, False]
    next_value = 2.0
    gamma = 0.9
    gae_lambda = 0.8

    # delta_1 = 2.0 + 0.9 * 2.0 - 1.0 = 2.8
    # adv_1 = 2.8
    # delta_0 = 1.0 + 0.9 * 1.0 - 1.0 = 0.9
    # adv_0 = 0.9 + 0.9 * 0.8 * 2.8 = 0.9 + 2.016 = 2.916
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    assert np.isclose(advantages[1], 2.8, atol=1e-4)
    assert np.isclose(advantages[0], 2.916, atol=1e-4)
    assert np.isclose(returns[1], 3.8, atol=1e-4)
    assert np.isclose(returns[0], 3.916, atol=1e-4)
