from pathlib import Path

import numpy as np
from src.environment.env import TicketToRideEnv
from src.rl.alphazero_trainer import AlphaZeroTrainer, SelfPlayReplayBuffer
from src.rl.policy_value_net import PolicyValueNetwork


def test_self_play_replay_buffer():
    buf = SelfPlayReplayBuffer(capacity=100)
    obs_dim = 10
    action_dim = 5

    for _ in range(20):
        obs = np.random.randn(obs_dim).astype(np.float32)
        mask = np.ones(action_dim, dtype=np.float32)
        pi = np.full(action_dim, 0.2, dtype=np.float32)
        z = 1.0
        buf.add(obs, mask, pi, z)

    assert len(buf) == 20
    b_obs, b_mask, b_pi, b_z = buf.sample(batch_size=8)
    assert b_obs.shape == (8, obs_dim)
    assert b_mask.shape == (8, action_dim)
    assert b_pi.shape == (8, action_dim)
    assert b_z.shape == (8, 1)


def test_alphazero_trainer_collect_and_train(tmp_path: Path):
    env = TicketToRideEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net = PolicyValueNetwork(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, num_res_blocks=1
    )
    trainer = AlphaZeroTrainer(net=net, num_simulations=5, batch_size=16)

    # Collect 1 self-play game
    num_steps = trainer.collect_self_play_games(num_games=1, max_turns=10)
    assert num_steps > 0
    assert len(trainer.buffer) == num_steps

    # Train 1 step
    metrics = trainer.train_step(batch_size=4)
    assert "loss" in metrics
    assert "value_loss" in metrics
    assert "policy_loss" in metrics
    assert metrics["loss"] > 0.0

    # Test saving
    ckpt_path = tmp_path / "az_ckpt.pt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()
