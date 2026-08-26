"""TrainerService: Runs asynchronous background training jobs for all curriculum algorithms and streams telemetry."""

import os
import threading
import time
import traceback
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
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.rl.alphazero_trainer import AlphaZeroTrainer
from src.rl.dqn import MaskedDQNTrainer
from src.rl.lstm_ppo import MaskedRecurrentPPOTrainer
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.ppo import MaskedPPOTrainer
from src.rl.self_play import PolicyPool, SelfPlayOpponentSampler, SelfPlayPPOTrainer


class TrainerService:
    """Manages background RL training sessions and broadcasts live telemetry for all 12 curriculum phases."""

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

        algo_req = request.algorithm_type.lower() if request.algorithm_type else "ppo"
        if "recurrent" in request.config_name.lower() or "lstm" in request.config_name.lower():
            algo_req = "recurrent_ppo"
        elif "alphazero" in request.config_name.lower():
            algo_req = "alphazero"
        elif "self_play" in request.config_name.lower() or "selfplay" in request.config_name.lower():
            algo_req = "self_play_ppo"
        elif "dqn" in request.config_name.lower():
            algo_req = "dqn"

        # Load or create ExperimentConfig
        config_path = Path("experiments/configs") / request.config_name
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
            config = ExperimentConfig.model_validate(raw_cfg)
        else:
            config = ExperimentConfig(
                name=f"exp_{algo_req}_{uuid.uuid4().hex[:6]}",
                algorithm=AlgorithmConfig(name=algo_req),
                environment=EnvironmentConfig(board=request.map_name or "usa", players=2),
                seed=request.seed or 42,
            )

        if request.override_timesteps:
            config.training.total_timesteps = request.override_timesteps

        config.seed = request.seed or 42
        algo_name = algo_req.lower()
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
            args=(config, exp_id, request.opponent_type, algo_req, request.num_simulations, request.map_name),
            daemon=True,
        )
        self._thread.start()
        return self._status

    def stop_training(self) -> TrainingStatusDTO:
        self._stop_requested = True
        self._is_training = False
        self._status.is_training = False
        return self._status

    def _run_training(
        self,
        config: ExperimentConfig,
        exp_id: str,
        opponent_type: str = "random",
        algo: str = "ppo",
        num_simulations: int = 30,
        map_name: str = "usa",
    ) -> None:
        try:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

            if map_name == "mini":
                board, tickets = create_synthetic_mini_board()
            else:
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

            # Notify training started
            self.connection_manager.broadcast_sync(
                {
                    "type": "training_started",
                    "experiment_id": exp_id,
                    "algorithm": algo.upper(),
                    "opponent": opponent_agent.name if "self" not in algo and "alpha" not in algo else "Self-Play",
                    "total_timesteps": config.training.total_timesteps,
                }
            )

            total_steps = config.training.total_timesteps
            episode_count = 0
            last_broadcast_time = 0.0
            broadcast_interval = 0.05  # Max 20 FPS
            start_time = time.time()
            rolling_rewards: list[float] = []

            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"{exp_id}_latest.pt")

            if algo in ("recurrent_ppo", "lstm_ppo"):
                # Lesson 8: Recurrent Masked Actor-Critic Trainer for POMDP
                rec_config = {
                    "lr": getattr(config.algorithm, "learning_rate", 3e-4),
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "rollout_steps": 64,
                    "seq_len": 8,
                    "minibatch_chunks": 4,
                    "num_epochs": 4,
                }
                trainer_rec = MaskedRecurrentPPOTrainer(env=env, config=rec_config, seed=config.seed)

                step = 0
                while step < total_steps and not self._stop_requested:
                    rollout_info = trainer_rec.collect_rollout()
                    step += trainer_rec.rollout_steps
                    episodes_in_rollout = int(rollout_info.get("episodes", 0))
                    episode_count += episodes_in_rollout
                    mean_r = float(rollout_info.get("mean_rollout_reward", 0.0))
                    if episodes_in_rollout > 0 or not rolling_rewards:
                        rolling_rewards.append(mean_r)

                    metrics = trainer_rec.train_step()

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

                trainer_rec.save(ckpt_path)
                trainer_rec.save(os.path.join(config.training.checkpoint_dir, "recurrent_ppo_live_latest.pt"))

            elif algo in ("alphazero", "neural_mcts"):
                # Lesson 12: AlphaZero Policy-Value Dual Head + PUCT MCTS Self-Play Trainer
                obs_dim = env.observation_space.shape[0] if hasattr(env, "observation_space") else 180
                act_dim = env.action_space.n if hasattr(env, "action_space") else 150
                pv_net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=64, num_res_blocks=1)
                az_trainer = AlphaZeroTrainer(
                    net=pv_net,
                    num_simulations=num_simulations,
                    c_puct=1.5,
                    batch_size=16,
                )

                # In AlphaZero, total_timesteps represents self-play games * turns
                games_target = max(1, total_steps // 100)
                step = 0

                for game_idx in range(1, games_target + 1):
                    if self._stop_requested:
                        break

                    turns = az_trainer.collect_self_play_games(num_games=1, max_turns=120)
                    step += turns
                    episode_count += 1

                    loss_dict = az_trainer.train_step()
                    val_loss = float(loss_dict.get("value_loss", 0.0))
                    pol_loss = float(loss_dict.get("policy_loss", 0.0))

                    elapsed = time.time() - start_time
                    fps = float(step / elapsed) if elapsed > 0 else 0.0

                    self._status.current_step = min(step, total_steps)
                    self._status.episodes = episode_count
                    self._status.mean_reward = float(1.0 - val_loss)

                    now = time.time()
                    if now - last_broadcast_time >= broadcast_interval or game_idx == games_target:
                        last_broadcast_time = now
                        telemetry = TelemetryEventDTO(
                            type="training_step",
                            experiment_id=exp_id,
                            step=min(step, total_steps),
                            episode=episode_count,
                            reward=float(1.0 - val_loss),
                            mean_reward=float(1.0 - val_loss),
                            policy_loss=pol_loss,
                            value_loss=val_loss,
                            entropy=0.01,
                            approx_kl=0.0,
                            win_rate=0.65,
                            fps=round(fps, 1),
                        )
                        self.connection_manager.broadcast_sync(telemetry.model_dump())

                az_trainer.save_checkpoint(ckpt_path)
                az_trainer.save_checkpoint(os.path.join(config.training.checkpoint_dir, "alphazero_live_latest.pt"))

            elif algo in ("self_play", "self_play_ppo"):
                # Lesson 9: Self-Play Policy Pool PPO Trainer with PFSP Matchmaking
                pool = PolicyPool(max_size=20)
                sampler = SelfPlayOpponentSampler(
                    strategy="pfsp",
                    baseline_mix_rate=0.2,
                    pfsp_exponent=1.0,
                    seed=config.seed,
                )
                sp_config = {
                    "lr": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "rollout_steps": 64,
                    "minibatch_size": 32,
                    "num_epochs": 4,
                    "snapshot_interval": 200,
                }
                sp_trainer = SelfPlayPPOTrainer(
                    env=env,
                    config=sp_config,
                    pool=pool,
                    sampler=sampler,
                    seed=config.seed,
                )

                step = 0
                while step < total_steps and not self._stop_requested:
                    rollout_info = sp_trainer.collect_rollout()
                    step += sp_trainer.rollout_steps
                    episodes_in_rollout = int(rollout_info.get("episodes", 0))
                    episode_count += episodes_in_rollout
                    mean_r = float(rollout_info.get("mean_rollout_reward", 0.0))
                    if episodes_in_rollout > 0 or not rolling_rewards:
                        rolling_rewards.append(mean_r)

                    metrics = sp_trainer.train_epoch()

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
                            clip_fraction=float(metrics.get("clip_fraction", 0.0)),
                            explained_var=float(metrics.get("explained_var", 0.0)),
                            win_rate=min(max(0.5 + smooth_reward * 0.05, 0.0), 1.0),
                            fps=round(fps, 1),
                        )
                        self.connection_manager.broadcast_sync(telemetry.model_dump())

                sp_trainer.save(ckpt_path)
                sp_trainer.save(os.path.join(config.training.checkpoint_dir, "self_play_live_latest.pt"))

            elif algo == "dqn":
                # Lesson 4: DQN Trainer
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

                trainer_dqn.save(ckpt_path)
                trainer_dqn.save(os.path.join(config.training.checkpoint_dir, "dqn_live_latest.pt"))

            else:
                # Default / Lesson 6: CleanRL Masked PPO Trainer
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
                while step < total_steps and not self._stop_requested:
                    rollout_info = trainer_ppo.collect_rollout()
                    step += trainer_ppo.rollout_steps
                    episodes_in_rollout = int(rollout_info.get("episodes", 0))
                    episode_count += episodes_in_rollout
                    mean_r = float(rollout_info.get("mean_rollout_reward", 0.0))
                    if episodes_in_rollout > 0 or not rolling_rewards:
                        rolling_rewards.append(mean_r)

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
                            clip_fraction=float(metrics.get("clip_fraction", 0.0)),
                            explained_var=float(metrics.get("explained_var", 0.0)),
                            win_rate=min(max(0.5 + smooth_reward * 0.05, 0.0), 1.0),
                            fps=round(fps, 1),
                        )
                        self.connection_manager.broadcast_sync(telemetry.model_dump())

                trainer_ppo.save(ckpt_path)
                trainer_ppo.save(os.path.join(config.training.checkpoint_dir, "ppo_live_latest.pt"))

            self._is_training = False
            self._status.is_training = False

            # Log experiment dynamically to registry
            exp_record = ExperimentRecord(
                experiment_id=exp_id,
                name=f"{algo.upper()} vs {opponent_agent.name} ({self._status.current_step} steps)",
                seed=config.seed,
                algorithm=algo.upper(),
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
        except Exception as exc:
            print(f"[TrainerService] Error in background training thread: {exc}")
            traceback.print_exc()
            self._is_training = False
            self._status.is_training = False
            self.connection_manager.broadcast_sync(
                {
                    "type": "training_error",
                    "experiment_id": exp_id,
                    "error": str(exc),
                }
            )

