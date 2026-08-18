"""Unit tests for RL networks, buffers, and advantage calculation."""

import numpy as np
import torch
from src.rl.advantage import compute_gae
from src.rl.networks import ActorCriticMLP, QNetworkMLP
from src.rl.replay_buffer import ReplayBuffer
from src.rl.self_play import PolicyPool


def test_q_network_mlp():
    net = QNetworkMLP(input_dim=16, action_dim=4, hidden_dim=32)
    x = torch.randn(2, 16)
    out = net(x)
    assert out.shape == (2, 4)


def test_actor_critic_mlp():
    net = ActorCriticMLP(input_dim=16, action_dim=4, hidden_dim=32)
    x = torch.randn(2, 16)
    logits, value = net(x)
    assert logits.shape == (2, 4)
    assert value.shape == (2, 1)


def test_replay_buffer():
    buf = ReplayBuffer(capacity=10)
    buf.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
    assert len(buf) == 1
    obs, _acts, _rews, _next_obs, _dones = buf.sample(1)
    assert obs.shape == (1, 4)


def test_compute_gae():
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    dones = [False, False, True]
    next_value = 0.0

    advantages, returns = compute_gae(rewards, values, dones, next_value)
    assert len(advantages) == 3
    assert len(returns) == 3


def test_policy_pool():
    pool = PolicyPool(max_size=3)
    pool.add_checkpoint("ckpt_1.pt")
    pool.add_checkpoint("ckpt_2.pt")
    sampled = pool.sample_opponent()
    assert sampled in ["ckpt_1.pt", "ckpt_2.pt"]
