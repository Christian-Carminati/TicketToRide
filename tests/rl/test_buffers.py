import numpy as np
import torch

from src.rl.replay_buffer import ReplayBuffer
from src.rl.rollout import RolloutBuffer


def test_replay_buffer_push_sample_tensors() -> None:
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        obs = np.ones(5, dtype=np.float32) * i
        action = i % 3
        reward = float(i)
        next_obs = np.ones(5, dtype=np.float32) * (i + 1)
        done = i == 9
        next_mask = np.array([1, 0, 1], dtype=np.int8)
        buffer.push(obs, action, reward, next_obs, done, next_mask)

    assert len(buffer) == 10
    batch = buffer.sample(batch_size=4)
    assert batch["obs"].shape == (4, 5)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["next_obs"].shape == (4, 5)
    assert batch["dones"].shape == (4,)
    assert batch["next_action_masks"].shape == (4, 3)
    assert batch["next_action_masks"].dtype == torch.bool


def test_rollout_buffer_minibatch_generator() -> None:
    rollout = RolloutBuffer()
    for i in range(16):
        rollout.add(
            obs=np.ones(4, dtype=np.float32) * i,
            action=i % 2,
            reward=1.0,
            value=0.5,
            log_prob=-0.69,
            done=False,
            action_mask=np.array([1, 1], dtype=np.int8),
        )

    assert len(rollout) == 16
    advantages = np.ones(16, dtype=np.float32) * 2.0
    returns = np.ones(16, dtype=np.float32) * 3.0

    minibatches = list(rollout.generate_minibatches(batch_size=4, advantages=advantages, returns=returns))
    assert len(minibatches) == 4
    for mb in minibatches:
        assert mb["obs"].shape == (4, 4)
        assert mb["actions"].shape == (4,)
        assert mb["advantages"].shape == (4,)
        assert mb["returns"].shape == (4,)
        assert mb["action_masks"].shape == (4, 2)

    rollout.clear()
    assert len(rollout) == 0
