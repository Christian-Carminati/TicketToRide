import numpy as np
import pytest
import torch

from src.rl.rollout import RecurrentRolloutBuffer


def test_recurrent_rollout_buffer_add_and_chunking():
    capacity = 32
    obs_dim = 10
    action_dim = 4
    lstm_hidden_dim = 16
    buf = RecurrentRolloutBuffer(
        capacity=capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lstm_hidden_dim=lstm_hidden_dim,
    )

    for i in range(capacity):
        obs = np.ones(obs_dim, dtype=np.float32) * i
        action = i % action_dim
        reward = 1.0
        val = float(i)
        lp = -0.5
        done = (i % 8 == 7)
        mask = np.ones(action_dim, dtype=bool)
        h = np.ones(lstm_hidden_dim, dtype=np.float32) * i
        c = np.ones(lstm_hidden_dim, dtype=np.float32) * i
        buf.add(obs, action, reward, val, lp, done, mask, h, c)

    assert buf.size == capacity

    # Set mock advantages and returns
    advs = np.linspace(0, 1, capacity, dtype=np.float32)
    rets = np.linspace(1, 2, capacity, dtype=np.float32)
    buf.set_advantages_and_returns(advs, rets)

    chunks = list(buf.generate_recurrent_chunks(seq_len=8, batch_size=2, num_epochs=1))
    assert len(chunks) == 2  # 32 / (8 * 2) = 2 batches

    batch = chunks[0]
    assert batch["obs"].shape == (2, 8, obs_dim)
    assert batch["actions"].shape == (2, 8)
    assert batch["old_log_probs"].shape == (2, 8)
    assert batch["values"].shape == (2, 8)
    assert batch["advantages"].shape == (2, 8)
    assert batch["returns"].shape == (2, 8)
    assert batch["action_masks"].shape == (2, 8, action_dim)
    assert batch["dones"].shape == (2, 8)
    assert batch["initial_h"].shape == (1, 2, lstm_hidden_dim)
    assert batch["initial_c"].shape == (1, 2, lstm_hidden_dim)


def test_recurrent_rollout_buffer_chunk_continuity_and_hidden_alignment():
    capacity = 16
    obs_dim = 4
    action_dim = 2
    lstm_hidden_dim = 8
    buf = RecurrentRolloutBuffer(
        capacity=capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lstm_hidden_dim=lstm_hidden_dim,
    )

    for i in range(capacity):
        obs = np.full(obs_dim, i, dtype=np.float32)
        h = np.full(lstm_hidden_dim, i * 10, dtype=np.float32)
        c = np.full(lstm_hidden_dim, i * 10 + 1, dtype=np.float32)
        mask = np.ones(action_dim, dtype=bool)
        buf.add(obs, i % action_dim, 0.5, float(i), -0.1, False, mask, h, c)

    advs = np.arange(capacity, dtype=np.float32)
    rets = np.arange(capacity, dtype=np.float32)
    buf.set_advantages_and_returns(advs, rets)

    # 16 timesteps, seq_len=4 -> 4 chunks. batch_size=4 -> 1 batch containing all 4 chunks in some shuffled order
    batches = list(buf.generate_recurrent_chunks(seq_len=4, batch_size=4, num_epochs=1))
    assert len(batches) == 1
    batch = batches[0]
    assert batch["obs"].shape == (4, 4, obs_dim)

    # Verify that each chunk of 4 steps is internally contiguous and initial_h matches step 0 of that chunk
    for b in range(4):
        chunk_obs = batch["obs"][b].numpy()  # shape (4, obs_dim)
        start_val = chunk_obs[0, 0]
        # Chunk must start at 0, 4, 8, or 12
        assert start_val in [0.0, 4.0, 8.0, 12.0]
        # Timesteps in chunk must be sequential
        for t in range(4):
            assert chunk_obs[t, 0] == start_val + t
        # Initial hidden state must match timestep `start_val`
        assert batch["initial_h"][0, b, 0].item() == pytest.approx(start_val * 10)
        assert batch["initial_c"][0, b, 0].item() == pytest.approx(start_val * 10 + 1)


def test_recurrent_rollout_buffer_lazy_init_and_dynamic_growth():
    # Start with uninitialized obs/action dims and small capacity
    buf = RecurrentRolloutBuffer(capacity=4, obs_dim=0, action_dim=0, lstm_hidden_dim=8)
    assert not buf.initialized

    obs_dim = 6
    action_dim = 3
    for i in range(10):  # Exceeds initial capacity of 4
        obs = np.ones(obs_dim, dtype=np.float32) * i
        action = i % action_dim
        mask = np.ones(action_dim, dtype=bool)
        h = np.zeros(8, dtype=np.float32)
        c = np.zeros(8, dtype=np.float32)
        buf.add(obs, action, 1.0, 0.0, 0.0, False, mask, h, c)

    assert buf.initialized
    assert buf.size == 10
    assert buf.capacity >= 10
    assert buf.obs_buf.shape == (buf.capacity, obs_dim)
    assert buf.masks_buf.shape == (buf.capacity, action_dim)


def test_recurrent_rollout_buffer_clear():
    buf = RecurrentRolloutBuffer(capacity=8, obs_dim=2, action_dim=2, lstm_hidden_dim=4)
    for i in range(5):
        buf.add(
            np.zeros(2, dtype=np.float32),
            0,
            0.0,
            0.0,
            0.0,
            False,
            np.ones(2, dtype=bool),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )
    assert buf.size == 5
    buf.clear()
    assert buf.size == 0
    assert buf.ptr == 0


def test_recurrent_rollout_buffer_empty_or_undersized_chunks():
    buf = RecurrentRolloutBuffer(capacity=8, obs_dim=2, action_dim=2, lstm_hidden_dim=4)
    # 0 items
    chunks = list(buf.generate_recurrent_chunks(seq_len=4, batch_size=2))
    assert len(chunks) == 0

    # 3 items, seq_len=4 -> 0 chunks
    for i in range(3):
        buf.add(
            np.zeros(2, dtype=np.float32),
            0,
            0.0,
            0.0,
            0.0,
            False,
            np.ones(2, dtype=bool),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )
    chunks = list(buf.generate_recurrent_chunks(seq_len=4, batch_size=2))
    assert len(chunks) == 0


def test_recurrent_rollout_buffer_multi_epochs():
    buf = RecurrentRolloutBuffer(capacity=16, obs_dim=2, action_dim=2, lstm_hidden_dim=4)
    for i in range(16):
        buf.add(
            np.zeros(2, dtype=np.float32),
            0,
            0.0,
            0.0,
            0.0,
            False,
            np.ones(2, dtype=bool),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )
    buf.set_advantages_and_returns(np.zeros(16, dtype=np.float32), np.zeros(16, dtype=np.float32))

    chunks = list(buf.generate_recurrent_chunks(seq_len=4, batch_size=2, num_epochs=3))
    # 16 / 4 = 4 chunks per epoch. batch_size=2 -> 2 batches per epoch * 3 epochs = 6 batches
    assert len(chunks) == 6


def test_recurrent_rollout_buffer_imports():
    from src.rl import RecurrentRolloutBuffer as Buf1
    from src.rl.lstm_ppo import RecurrentRolloutBuffer as Buf2
    from src.rl.rollout import RecurrentRolloutBuffer as Buf3

    assert Buf1 is Buf3
    assert Buf2 is Buf3

