import os
import tempfile
import numpy as np
import pytest
import torch

from src.environment.env import TicketToRideEnv
from src.rl.lstm_ppo import MaskedRecurrentPPOTrainer, RecurrentPPOTrainer
from src.rl import MaskedRecurrentPPOTrainer as ExportedTrainer, RecurrentPPOTrainer as ExportedAliasTrainer


def test_recurrent_ppo_trainer_train_step_execution():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 64,
            "seq_len": 8,
            "minibatch_chunks": 4,
            "num_epochs": 2,
            "hidden_dim": 32,
            "lstm_hidden_dim": 32,
        },
    )

    metrics = trainer.train_step()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "approx_kl" in metrics
    assert "mean_reward" in metrics
    assert "total_timesteps" in metrics
    assert not np.isnan(metrics["policy_loss"])
    assert not np.isnan(metrics["value_loss"])
    assert not np.isnan(metrics["entropy"])
    assert trainer.total_timesteps == 64


def test_recurrent_ppo_trainer_deterministic_reproducibility():
    """Two trainers initialized with same seed must yield identical loss trajectories."""
    def run_trainer_run(seed=123):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = TicketToRideEnv(seed=seed)
        trainer = MaskedRecurrentPPOTrainer(
            env=env,
            config={
                "rollout_steps": 64,
                "seq_len": 8,
                "minibatch_chunks": 4,
                "num_epochs": 2,
                "hidden_dim": 32,
                "lstm_hidden_dim": 32,
            },
        )
        return trainer.train_step()

    metrics1 = run_trainer_run(123)
    metrics2 = run_trainer_run(123)

    assert metrics1["policy_loss"] == pytest.approx(metrics2["policy_loss"], rel=1e-5)
    assert metrics1["value_loss"] == pytest.approx(metrics2["value_loss"], rel=1e-5)
    assert metrics1["entropy"] == pytest.approx(metrics2["entropy"], rel=1e-5)
    assert metrics1["approx_kl"] == pytest.approx(metrics2["approx_kl"], rel=1e-5)


def test_recurrent_ppo_trainer_save_and_load():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 1,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
        },
    )
    trainer.train_step()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        trainer.save(tmp_path)
        assert os.path.exists(tmp_path)

        # Create a new trainer and load the saved checkpoint
        new_trainer = MaskedRecurrentPPOTrainer(
            env=TicketToRideEnv(seed=42),
            config={
                "rollout_steps": 32,
                "seq_len": 8,
                "minibatch_chunks": 2,
                "num_epochs": 1,
                "hidden_dim": 16,
                "lstm_hidden_dim": 16,
            },
        )
        new_trainer.load(tmp_path)

        assert new_trainer.total_timesteps == trainer.total_timesteps
        for p1, p2 in zip(trainer.actor_critic.parameters(), new_trainer.actor_critic.parameters()):
            assert torch.equal(p1, p2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_recurrent_ppo_trainer_legacy_checkpoint_load():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 1,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
        },
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(
            {
                "actor_critic_state_dict": trainer.actor_critic.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "total_timesteps": 128,
            },
            tmp_path,
        )

        new_trainer = MaskedRecurrentPPOTrainer(
            env=TicketToRideEnv(seed=42),
            config={
                "rollout_steps": 32,
                "seq_len": 8,
                "minibatch_chunks": 2,
                "num_epochs": 1,
                "hidden_dim": 16,
                "lstm_hidden_dim": 16,
            },
        )
        new_trainer.load(tmp_path)
        assert new_trainer.total_timesteps == 128
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_recurrent_ppo_trainer_lr_annealing_and_train_loop():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 1,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
            "lr": 1e-3,
            "anneal_lr": True,
        },
    )

    callback_records = []
    logs = trainer.train(total_timesteps=64, callback=lambda m: callback_records.append(m))

    assert len(logs) == 2
    assert len(callback_records) == 2
    assert trainer.total_timesteps >= 64
    assert trainer.lr < 1e-3


def test_recurrent_ppo_trainer_clip_vloss_option():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 2,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
            "clip_vloss": True,
            "vf_clip_eps": 0.1,
        },
    )
    metrics = trainer.train_step()
    assert "value_loss" in metrics
    assert not np.isnan(metrics["value_loss"])


def test_recurrent_ppo_trainer_norm_adv_false():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 1,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
            "norm_adv": False,
        },
    )
    metrics = trainer.train_step()
    assert "policy_loss" in metrics
    assert not np.isnan(metrics["policy_loss"])


def test_recurrent_ppo_trainer_target_kl_early_stopping():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 10,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
            "target_kl": 0.00001,  # Strict threshold triggers early stopping
        },
    )
    metrics = trainer.train_step()
    assert "approx_kl" in metrics


def test_recurrent_ppo_trainer_collect_rollout():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "seq_len": 8,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
        },
    )
    stats = trainer.collect_rollout()
    assert "mean_reward" in stats
    assert "episodes_completed" in stats
    assert trainer.rollout_buffer.size == 32
    assert trainer.total_timesteps == 32


def test_recurrent_ppo_trainer_undersized_buffer():
    env = TicketToRideEnv(seed=42)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 4,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 1,
            "hidden_dim": 16,
            "lstm_hidden_dim": 16,
        },
    )
    metrics = trainer.train_step()
    assert "mean_reward" in metrics
    assert "policy_loss" not in metrics


def test_recurrent_ppo_trainer_export_import():
    assert ExportedTrainer is MaskedRecurrentPPOTrainer
    assert ExportedAliasTrainer is RecurrentPPOTrainer
    assert RecurrentPPOTrainer is MaskedRecurrentPPOTrainer
