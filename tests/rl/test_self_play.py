import pytest
import torch
from pathlib import Path
from src.rl.networks import MaskedActorCritic
from src.rl.lstm_ppo import RecurrentMaskedActorCritic
from src.rl.self_play import PolicyPool, PolicySnapshot
from src.agents.ppo_agent import PPOAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent


def test_policy_pool_add_and_retrieve():
    pool = PolicyPool(max_size=5)
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)

    snap0 = pool.add_policy(model, step=0, name="gen_0", metadata={"loss": 1.0})
    assert snap0.generation == 0
    assert snap0.step == 0
    assert snap0.name == "gen_0"
    assert not snap0.is_recurrent
    assert pool.size == 1

    # Retrieve by index and by name
    assert pool.get_snapshot(0).name == "gen_0"
    assert pool.get_snapshot("gen_0").step == 0


def test_policy_pool_capacity_and_retention():
    pool = PolicyPool(max_size=3)
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)

    for i in range(5):
        pool.add_policy(model, step=i * 1000, name=f"gen_{i}")

    assert pool.size == 3
    # Initial generation gen_0 must always be preserved as anchor
    assert pool.get_snapshot("gen_0") is not None
    # Latest generation must be present
    assert pool.get_snapshot("gen_4") is not None


def test_policy_pool_create_agent():
    pool = PolicyPool()
    mlp_model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    lstm_model = RecurrentMaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32, lstm_hidden_dim=32)

    pool.add_policy(mlp_model, step=100, name="mlp_snap")
    pool.add_policy(lstm_model, step=200, name="lstm_snap")

    agent_mlp = pool.create_agent("mlp_snap")
    assert isinstance(agent_mlp, PPOAgent)

    agent_lstm = pool.create_agent("lstm_snap")
    assert isinstance(agent_lstm, RecurrentPPOAgent)


def test_policy_pool_save_and_load(tmp_path: Path):
    pool = PolicyPool()
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    pool.add_policy(model, step=500, name="gen_saved")

    save_dir = str(tmp_path / "pool_export")
    pool.save_pool(save_dir)

    new_pool = PolicyPool()
    new_pool.load_pool(save_dir)
    assert new_pool.size == 1
    assert new_pool.get_snapshot(0).name == "gen_saved"
