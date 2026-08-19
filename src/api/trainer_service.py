"""TrainerService: Runs asynchronous background training jobs and streams telemetry."""

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.api.schemas import TelemetryEventDTO, TrainingStartRequest, TrainingStatusDTO
from src.api.websocket import ConnectionManager
from src.environment.env import TicketToRideEnv
from src.experiments.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig
from src.experiments.registry import ExperimentRecord, ExperimentRegistry
from src.game.maps import load_usa_board
from src.rl.dqn import MaskedDQNTrainer
from src.rl.ppo import MaskedPPOTrainer


class TrainerService:
    """Manages background RL training sessions and broadcasts live telemetry."""

    def __init__(self, connection_manager: ConnectionManager | None = None) -> None:
        self.connection_manager = connection_manager or ConnectionManager()
        self._is_training = False
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._status = TrainingStatusDTO(is_training=False)
        self.registry = ExperimentRegistry()

    def get_status(self) -> TrainingStatusDTO:
        return self._status

    def start_training(self, request: TrainingStartRequest) -> TrainingStatusDTO:
        if self._is_training:
            return self._status

        # Load or create ExperimentConfig
        config_path = Path("experiments/configs") / request.config_name
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
            config = ExperimentConfig.model_validate(raw_cfg)
        else:
            algo_req = "dqn" if "dqn" in request.config_name.lower() else "ppo"
            config = ExperimentConfig(
                name=f"exp_{uuid.uuid4().hex[:6]}",
                algorithm=AlgorithmConfig(name=algo_req),
                environment=EnvironmentConfig(board="usa", players=2),
                seed=request.seed or 42,
            )

        if request.override_timesteps:
            config.training.total_timesteps = request.override_timesteps

        config.seed = request.seed or 42
        algo_name = config.algorithm.name if hasattr(config.algorithm, "name") else str(config.algorithm)
        exp_id = f"{config.name}_{uuid.uuid4().hex[:4]}"

        self._is_training = True
        self._stop_requested = False
        self._status = TrainingStatusDTO(
            is_training=True,
            experiment_id=exp_id,
            algorithm=algo_name,
            current_step=0,
            total_timesteps=config.training.total_timesteps,
            episodes=0,
            mean_reward=0.0,
        )

        self._thread = threading.Thread(
            target=self._run_training,
            args=(config, exp_id, request.opponent_type),
            daemon=True,
        )
        self._thread.start()
        return self._status

    def stop_training(self) -> TrainingStatusDTO:
        self._stop_requested = True
        self._is_training = False
        self._status.is_training = False
        return self._status

    def _run_training(self, config: ExperimentConfig, exp_id: str, opponent_type: str = "random") -> None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        board, tickets = load_usa_board()

        opp_clean = opponent_type.lower()
        if opp_clean == "greedy":
            opponent_agent = GreedyAgent(name="GreedyBot")
        elif opp_clean == "strategic":
            opponent_agent = StrategicAgent(name="StrategicBot")
        else:
            opponent_agent = RandomAgent(name="RandomBot", seed=config.seed)

        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=opponent_agent,
            num_players=config.environment.players,
        )

        algo_name = config.algorithm.name if hasattr(config.algorithm, "name") else str(config.algorithm)
        algo = algo_name.lower()

        # Notify training started
        self.connection_manager.broadcast_sync(
            {
                "type": "training_started",
                "experiment_id": exp_id,
                "algorithm": algo_name,
                "opponent": opponent_agent.name,
                "total_timesteps": config.training.total_timesteps,
            }
        )

        total_steps = config.training.total_timesteps
        episode_count = 0
        last_broadcast_time = 0.0
        broadcast_interval = 0.05  # Max 20 FPS

        if algo == "ppo":
            ppo_config = {
                "lr": getattr(config.algorithm, "learning_rate", 3e-4),
                "gamma": getattr(config.algorithm, "gamma", 0.99),
                "gae_lambda": getattr(config.algorithm, "gae_lambda", 0.95),
                "rollout_steps": min(getattr(config.training, "rollout_steps", 64), 64),
                "minibatch_size": getattr(config.training, "batch_size", 32),
                "num_epochs": getattr(config.training, "num_epochs", 4),
                "clip_eps": getattr(config.algorithm, "clip_range", 0.2),
                "ent_coef": getattr(config.algorithm, "entropy_coef", 0.01),
                "vf_coef": getattr(config.algorithm, "value_coef", 0.5),
            }
            trainer_ppo = MaskedPPOTrainer(env=env, config=ppo_config)

            step = 0
            rolling_rewards: list[float] = []
            start_time = time.time()

            while step < total_steps and not self._stop_requested:
                # 1. Collect rollout
                rollout_info = trainer_ppo.collect_rollout()
                step += trainer_ppo.rollout_steps
                episodes_in_rollout = int(rollout_info.get("episodes", 0))
                episode_count += episodes_in_rollout
                mean_r = float(rollout_info.get("mean_rollout_reward", 0.0))
                if episodes_in_rollout > 0 or not rolling_rewards:
                    rolling_rewards.append(mean_r)

                # 2. Train epoch
                metrics = trainer_ppo.train_epoch()

                elapsed = time.time() - start_time
                fps = float(step / elapsed) if elapsed > 0 else 0.0
                smooth_reward = float(np.mean(rolling_rewards[-10:])) if rolling_rewards else mean_r

                self._status.current_step = min(step, total_steps)
                self._status.episodes = episode_count
                self._status.mean_reward = smooth_reward

                now = time.time()
                if now - last_broadcast_time >= broadcast_interval or step >= total_steps:
                    last_broadcast_time = now
                    telemetry = TelemetryEventDTO(
                        type="training_step",
                        experiment_id=exp_id,
                        step=min(step, total_steps),
                        episode=episode_count,
                        reward=mean_r,
                        mean_reward=smooth_reward,
                        policy_loss=float(metrics.get("policy_loss", 0.0)),
                        value_loss=float(metrics.get("value_loss", 0.0)),
                        entropy=float(metrics.get("entropy", 0.0)),
                        approx_kl=float(metrics.get("approx_kl", 0.0)),
                        win_rate=min(max(0.5 + smooth_reward * 0.05, 0.0), 1.0),
                        fps=round(fps, 1),
                    )
                    self.connection_manager.broadcast_sync(telemetry.model_dump())

            # Auto-save checkpoints
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"{exp_id}_latest.pt")
            live_path = os.path.join(config.training.checkpoint_dir, "ppo_live_latest.pt")
            trainer_ppo.save(ckpt_path)
            trainer_ppo.save(live_path)

        else:
            # DQN Trainer
            dqn_config = {
                "lr": getattr(config.algorithm, "learning_rate", 5e-4),
                "gamma": getattr(config.algorithm, "gamma", 0.99),
                "batch_size": getattr(config.algorithm, "batch_size", 32),
                "buffer_size": getattr(config.algorithm, "buffer_size", 10000),
                "target_update_freq": getattr(config.algorithm, "target_update_freq", 200),
                "epsilon_start": getattr(config.algorithm, "epsilon_start", 1.0),
                "epsilon_end": getattr(config.algorithm, "epsilon_end", 0.05),
                "epsilon_decay_steps": getattr(config.algorithm, "epsilon_decay_steps", 5000),
                "learning_starts": getattr(config.algorithm, "learning_starts", 50),
            }
            trainer_dqn = MaskedDQNTrainer(env=env, config=dqn_config)

            episode_rewards: list[float] = []
            current_ep_reward = 0.0
            start_time = time.time()

            for step in range(1, total_steps + 1):
                if self._stop_requested:
                    break

                reward, done = trainer_dqn.step()
                current_ep_reward += reward

                if done:
                    episode_count += 1
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

                metrics = trainer_dqn.train_step()

                now = time.time()
                if now - last_broadcast_time >= broadcast_interval or step == total_steps or done:
                    last_broadcast_time = now
                    mean_r = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
                    elapsed = time.time() - start_time
                    fps = float(step / elapsed) if elapsed > 0 else 0.0

                    self._status.current_step = step
                    self._status.episodes = episode_count
                    self._status.mean_reward = mean_r

                    telemetry = TelemetryEventDTO(
                        type="training_step",
                        experiment_id=exp_id,
                        step=step,
                        episode=episode_count,
                        reward=float(reward),
                        mean_reward=mean_r,
                        policy_loss=float(metrics.get("loss", 0.0)),
                        value_loss=float(metrics.get("loss", 0.0)),
                        entropy=float(metrics.get("epsilon", 0.0)),
                        approx_kl=0.0,
                        win_rate=min(max(0.5 + mean_r * 0.05, 0.0), 1.0),
                        fps=round(fps, 1),
                    )
                    self.connection_manager.broadcast_sync(telemetry.model_dump())

            # Auto-save checkpoints
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"{exp_id}_latest.pt")
            live_path = os.path.join(config.training.checkpoint_dir, "dqn_live_latest.pt")
            trainer_dqn.save(ckpt_path)
            trainer_dqn.save(live_path)

        self._is_training = False
        self._status.is_training = False

        # Log experiment dynamically to registry
        exp_record = ExperimentRecord(
            experiment_id=exp_id,
            name=f"{algo_name.upper()} vs {opponent_agent.name} ({self._status.current_step} steps)",
            seed=config.seed,
            algorithm=algo_name,
            env_version=1,
            reward_version=1,
            metrics={
                f"mean_reward_vs_{opp_clean}": round(self._status.mean_reward, 2),
                f"episodes_vs_{opp_clean}": episode_count,
                "total_timesteps": self._status.current_step,
            },
        )
        self.registry.log_experiment(exp_record)

        self.connection_manager.broadcast_sync(
            {
                "type": "training_finished",
                "experiment_id": exp_id,
                "total_steps": self._status.current_step,
                "episodes": episode_count,
            }
        )
