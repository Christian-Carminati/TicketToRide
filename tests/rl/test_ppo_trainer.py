import os
import tempfile

import numpy as np
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.game.maps import create_synthetic_mini_board
from src.rl.ppo import MaskedPPOTrainer


def test_ppo_trainer_rollout_and_train_epoch() -> None:
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))

    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "rollout_steps": 64,
        "num_epochs": 2,
        "minibatch_size": 16,
    }

    trainer = MaskedPPOTrainer(env=env, config=config)

    trainer.collect_rollout()
    assert len(trainer.rollout_buffer) == 64

    train_metrics = trainer.train_epoch()
    assert "policy_loss" in train_metrics
    assert "value_loss" in train_metrics
    assert "entropy" in train_metrics
    assert "approx_kl" in train_metrics
    assert "explained_var" in train_metrics
    assert np.isfinite(train_metrics["policy_loss"])


def test_ppo_trainer_save_and_load() -> None:
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))

    config = {
        "rollout_steps": 32,
        "num_epochs": 1,
        "minibatch_size": 16,
    }

    trainer = MaskedPPOTrainer(env=env, config=config)
    trainer.collect_rollout()
    trainer.train_epoch()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ppo_test.pt")
        trainer.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        new_trainer = MaskedPPOTrainer(env=env, config=config)
        new_trainer.load(ckpt_path)
        assert new_trainer.total_timesteps == trainer.total_timesteps
