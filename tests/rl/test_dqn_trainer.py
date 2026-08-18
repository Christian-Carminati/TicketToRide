import os
import tempfile

import numpy as np
import torch

from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.game.maps import create_synthetic_mini_board
from src.rl.dqn import MaskedDQNTrainer


def test_dqn_trainer_initialization_and_train_step() -> None:
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))

    config = {
        "lr": 1e-3,
        "gamma": 0.99,
        "buffer_size": 1000,
        "batch_size": 16,
        "target_update_freq": 50,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 1000,
        "learning_starts": 10,
    }

    trainer = MaskedDQNTrainer(env=env, config=config)

    # Populate buffer with initial steps
    for _ in range(25):
        trainer.step(epsilon=1.0)

    assert len(trainer.replay_buffer) >= 25

    # Execute training step
    metrics = trainer.train_step()
    assert "loss" in metrics
    assert "q_mean" in metrics
    assert np.isfinite(metrics["loss"])


def test_dqn_trainer_epsilon_decay_and_save_load() -> None:
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))

    config = {
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 100,
        "learning_starts": 10,
        "batch_size": 8,
    }

    trainer = MaskedDQNTrainer(env=env, config=config)
    assert np.isclose(trainer.get_epsilon(), 1.0)

    for _ in range(100):
        trainer.step()

    assert np.isclose(trainer.get_epsilon(), 0.1, atol=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "dqn_test.pt")
        trainer.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        new_trainer = MaskedDQNTrainer(env=env, config=config)
        new_trainer.load(ckpt_path)
        assert new_trainer.total_timesteps == trainer.total_timesteps
