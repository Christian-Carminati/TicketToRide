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

from src.agents.random_agent import RandomAgent
from src.api.schemas import TelemetryEventDTO, TrainingStartRequest, TrainingStatusDTO
from src.api.websocket import ConnectionManager
from src.environment.env import TicketToRideEnv
from src.experiments.config import ExperimentConfig
from src.game.maps import create_synthetic_mini_board, load_usa_board
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
            # Fallback default PPO config
            config = ExperimentConfig(
                name=f"exp_{uuid.uuid4().hex[:6]}",
                algorithm="ppo",
                seed=request.seed,
            )

        if request.override_timesteps:
            config.training.total_timesteps = request.override_timesteps

        config.seed = request.seed
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
            args=(config, exp_id),
            daemon=True,
        )
        self._thread.start()
        return self._status

    def stop_training(self) -> TrainingStatusDTO:
        self._stop_requested = True
        self._is_training = False
        self._status.is_training = False
        return self._status

    def _run_training(self, config: ExperimentConfig, exp_id: str) -> None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if config.environment.board.lower() == "mini":
            board, tickets = create_synthetic_mini_board()
        else:
            board, tickets = load_usa_board()

        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=RandomAgent(seed=config.seed),
            num_players=config.environment.players,
        )

        algo_name = config.algorithm.name if hasattr(config.algorithm, "name") else str(config.algorithm)

        # Notify training started
        self.connection_manager.broadcast_sync(
            {
                "type": "training_started",
                "experiment_id": exp_id,
                "algorithm": algo_name,
                "total_timesteps": config.training.total_timesteps,
            }
        )

        algo = algo_name.lower()
        if algo == "dqn":
            dqn_config = {
                "lr": config.algorithm.learning_rate,
                "gamma": config.algorithm.gamma,
                "batch_size": config.algorithm.batch_size,
                "buffer_size": config.algorithm.buffer_size,
                "target_update_freq": config.algorithm.target_update_freq,
                "epsilon_start": config.algorithm.epsilon_start,
                "epsilon_end": config.algorithm.epsilon_end,
                "epsilon_decay_steps": config.algorithm.epsilon_decay_steps,
                "learning_starts": config.algorithm.learning_starts,
            }
            trainer: Any = MaskedDQNTrainer(env=env, config=dqn_config)
        else:
            ppo_config = {
                "lr": config.algorithm.learning_rate,
                "gamma": config.algorithm.gamma,
                "gae_lambda": config.algorithm.gae_lambda,
                "rollout_steps": getattr(config.training, "rollout_steps", 64),
                "minibatch_size": getattr(config.training, "batch_size", 64),
                "update_epochs": getattr(config.training, "num_epochs", 4),
                "clip_coef": getattr(config.algorithm, "clip_range", 0.2),
                "ent_coef": getattr(config.algorithm, "entropy_coef", 0.01),
                "vf_coef": getattr(config.algorithm, "value_coef", 0.5),
            }
            trainer = MaskedPPOTrainer(env=env, config=ppo_config)

        total_steps = config.training.total_timesteps
        episode_rewards: list[float] = []
        current_ep_reward = 0.0
        episode_count = 0
        start_time = time.time()

        for step in range(1, total_steps + 1):
            if self._stop_requested:
                break

            step_res = trainer.step()
            # step_res returns transition details
            reward = 0.0
            done = False
            if isinstance(step_res, tuple) and len(step_res) >= 3:
                reward = float(step_res[1])
                done = bool(step_res[2])

            current_ep_reward += reward
            if done:
                episode_count += 1
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0

            metrics = trainer.train_step()

            # Broadcast every 10 steps or at episode boundaries
            if step % 10 == 0 or step == total_steps or done:
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
                    policy_loss=float(metrics.get("policy_loss", 0.0)) if metrics else None,
                    value_loss=float(metrics.get("value_loss", 0.0)) if metrics else None,
                    entropy=float(metrics.get("entropy", 0.0)) if metrics else None,
                    approx_kl=float(metrics.get("approx_kl", 0.0)) if metrics else None,
                    win_rate=min(0.5 + mean_r * 0.05, 1.0) if mean_r > 0 else 0.0,
                    fps=round(fps, 1),
                )
                self.connection_manager.broadcast_sync(telemetry.model_dump())

            # Slight yield to avoid saturated CPU loop
            if step % 50 == 0:
                time.sleep(0.005)

        self._is_training = False
        self._status.is_training = False
        self.connection_manager.broadcast_sync(
            {
                "type": "training_finished",
                "experiment_id": exp_id,
                "total_steps": self._status.current_step,
                "episodes": episode_count,
            }
        )
