import numpy as np
from src.environment.env import TicketToRideEnv
from src.rl.ppo import MaskedPPOTrainer


def test_ppo_learning_rate_annealing() -> None:
    env = TicketToRideEnv(board_type="mini")
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "lr": 1e-3,
            "anneal_lr": True,
            "rollout_steps": 100,
        },
    )
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    assert np.isclose(initial_lr, 1e-3)

    # Advance halfway through total_timesteps
    trainer.total_timesteps = 500
    trainer.update_learning_rate(total_timesteps=1000)
    current_lr = trainer.optimizer.param_groups[0]["lr"]
    assert np.isclose(current_lr, 5e-4, atol=1e-5)


def test_ppo_target_kl_early_stopping() -> None:
    env = TicketToRideEnv(board_type="mini")
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "target_kl": 1e-6,  # extraordinarily low threshold to guarantee early stopping
            "num_epochs": 10,
            "rollout_steps": 64,
            "minibatch_size": 16,
        },
    )
    trainer.collect_rollout()
    metrics = trainer.train_epoch()
    assert "approx_kl" in metrics
    assert "early_stopped" in metrics
    assert metrics["early_stopped"] is True
    assert metrics["epochs_completed"] < 10


def test_ppo_value_loss_clipping() -> None:
    env = TicketToRideEnv(board_type="mini")
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "clip_vloss": True,
            "vf_clip_eps": 0.2,
            "rollout_steps": 64,
            "minibatch_size": 16,
        },
    )
    trainer.collect_rollout()
    metrics = trainer.train_epoch()
    assert "value_loss" in metrics
    assert "clip_fraction" in metrics
    assert not np.isnan(metrics["value_loss"])
    assert np.isfinite(metrics["clip_fraction"])
