from src.api.schemas import TrainingStartRequest
from src.api.trainer_service import TrainerService
from src.environment.env import TicketToRideEnv
from src.game.maps import create_synthetic_mini_board
from src.rl.self_play import PolicyPool, SelfPlayOpponentSampler, SelfPlayPPOTrainer


def test_self_play_trainer_rollout_and_epoch():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, num_players=2)
    pool = PolicyPool(max_size=10)
    sampler = SelfPlayOpponentSampler(strategy="pfsp", seed=42)
    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "rollout_steps": 16,
        "minibatch_size": 8,
        "num_epochs": 2,
        "snapshot_interval": 50,
    }
    trainer = SelfPlayPPOTrainer(env=env, config=config, pool=pool, sampler=sampler, seed=42)

    rollout_info = trainer.collect_rollout()
    assert "mean_rollout_reward" in rollout_info
    metrics = trainer.train_epoch()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_trainer_service_self_play_startup():
    service = TrainerService()
    req = TrainingStartRequest(
        config_name="self_play_mini.yaml",
        algorithm_type="self_play_ppo",
        override_timesteps=64,
        map_name="mini",
        seed=42,
    )
    status = service.start_training(req)
    assert status.is_training is True
    assert status.algorithm == "self_play_ppo"
    service.stop_training()


def test_trainer_service_self_play_execution():
    service = TrainerService()
    TrainingStartRequest(
        config_name="self_play_mini.yaml",
        algorithm_type="self_play_ppo",
        override_timesteps=128,
        map_name="mini",
        seed=42,
    )
    # Run _run_training directly
    from src.experiments.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig

    cfg = ExperimentConfig(
        name="test_pfsp",
        algorithm=AlgorithmConfig(name="self_play_ppo"),
        environment=EnvironmentConfig(board="mini", players=2),
        seed=42,
    )
    cfg.training.total_timesteps = 128
    service._status.is_training = True
    service._run_training(
        config=cfg,
        exp_id="test_exp_pfsp",
        opponent_type="random",
        algo="self_play_ppo",
        num_simulations=10,
        map_name="mini",
    )
    assert service._status.current_step >= 64
    assert service._status.episodes >= 0
