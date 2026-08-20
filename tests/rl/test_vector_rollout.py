import numpy as np
import torch
from src.rl.advantage import compute_gae_vectorized
from src.rl.rollout import VectorRolloutBuffer


def test_vector_rollout_buffer_shapes_and_minibatches() -> None:
    num_steps = 16
    num_envs = 4
    obs_dim = 10
    action_dim = 5

    buf = VectorRolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    for _ in range(num_steps):
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        act = np.random.randint(0, action_dim, size=(num_envs,))
        rew = np.random.randn(num_envs).astype(np.float32)
        val = np.random.randn(num_envs).astype(np.float32)
        logp = np.random.randn(num_envs).astype(np.float32)
        done = np.zeros(num_envs, dtype=bool)
        masks = np.ones((num_envs, action_dim), dtype=bool)

        buf.add(
            obs=obs,
            action=act,
            reward=rew,
            value=val,
            log_prob=logp,
            done=done,
            action_mask=masks,
        )

    assert buf.step == num_steps
    assert buf.is_full()

    next_values = np.zeros(num_envs, dtype=np.float32)
    advs, returns = buf.compute_returns_and_advantages(
        next_values=next_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advs.shape == (num_steps, num_envs)
    assert returns.shape == (num_steps, num_envs)

    minibatches = list(
        buf.generate_minibatches(
            batch_size=16,
            advantages=advs,
            returns=returns,
            device="cpu",
        )
    )
    # Total samples = 16 * 4 = 64. Batch size = 16 => 4 minibatches
    assert len(minibatches) == 4
    mb0 = minibatches[0]
    assert mb0["obs"].shape == (16, obs_dim)
    assert mb0["actions"].shape == (16,)
    assert mb0["action_masks"].shape == (16, action_dim)
    assert mb0["advantages"].shape == (16,)
    assert mb0["returns"].shape == (16,)
    assert mb0["values"].shape == (16,)


def test_compute_gae_vectorized_terminal_masking() -> None:
    # 3 steps, 2 envs
    rewards = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]], dtype=np.float32)
    values = np.array([[0.5, 1.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32)
    dones = np.array([[False, False], [True, False], [False, False]], dtype=bool)
    next_values = np.array([1.0, 2.0], dtype=np.float32)

    advs, rets = compute_gae_vectorized(
        rewards=rewards,
        values=values,
        dones=dones,
        next_values=next_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advs.shape == (3, 2)
    assert rets.shape == (3, 2)
    assert np.all(np.isfinite(advs))
    assert np.all(np.isfinite(rets))
