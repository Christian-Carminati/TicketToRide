import os
import tempfile

import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent


def test_dqn_agent_inference_and_masking() -> None:
    agent = DQNAgent(name="TestDQN", input_dim=10, action_dim=5)
    obs = np.random.randn(10).astype(np.float32)
    mask = np.array([0, 1, 0, 1, 0], dtype=np.int8)

    action = agent.select_action(obs, action_mask=mask)
    assert action in [1, 3]


def test_ppo_agent_inference_and_masking() -> None:
    agent = PPOAgent(name="TestPPO", input_dim=10, action_dim=5)
    obs = np.random.randn(10).astype(np.float32)
    mask = np.array([1, 0, 0, 0, 0], dtype=np.int8)

    action = agent.select_action(obs, action_mask=mask)
    assert action == 0


def test_rl_agents_checkpoint_loading() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dqn_ckpt = os.path.join(tmpdir, "dqn.pt")
        ppo_ckpt = os.path.join(tmpdir, "ppo.pt")

        dqn1 = DQNAgent(input_dim=8, action_dim=4)
        torch.save({"policy_state_dict": dqn1.q_net.state_dict()}, dqn_ckpt)

        dqn2 = DQNAgent(input_dim=8, action_dim=4, model_path=dqn_ckpt)
        obs = np.ones(8, dtype=np.float32)
        mask = np.array([1, 1, 0, 0], dtype=np.int8)
        assert dqn1.select_action(obs, mask) == dqn2.select_action(obs, mask)

        ppo1 = PPOAgent(input_dim=8, action_dim=4)
        torch.save({"actor_critic_state_dict": ppo1.actor_critic.state_dict()}, ppo_ckpt)

        ppo2 = PPOAgent(input_dim=8, action_dim=4, model_path=ppo_ckpt)
        assert ppo1.select_action(obs, mask) == ppo2.select_action(obs, mask)
