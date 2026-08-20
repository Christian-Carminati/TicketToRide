"""Experiment execution engine for RL training and evaluation."""

import os
import uuid
from typing import Any

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.environment.reward import RewardFactory, RewardWeights
from src.experiments.config import ExperimentConfig
from src.experiments.evaluator import MultiOpponentEvaluator
from src.experiments.registry import ExperimentRecord, ExperimentRegistry
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.rl.dqn import MaskedDQNTrainer
from src.rl.ppo import MaskedPPOTrainer


class ExperimentRunner:
    """Orchestrates configuration-driven training, checkpointing, and multi-opponent evaluation."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.registry = ExperimentRegistry()

    def _setup_board(self):
        if self.config.environment.board.lower() == "mini":
            return create_synthetic_mini_board()
        return load_usa_board()

    def run(self) -> ExperimentRecord:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        board, tickets = self._setup_board()
        if self.config.environment.reward_config is not None:
            rc_dict = self.config.environment.reward_config.model_dump()
            v = rc_dict.pop("version", "custom")
            weights = RewardWeights(**{k: val for k, val in rc_dict.items() if hasattr(RewardWeights, k)})
            reward_calc = RewardFactory.create("custom", board=board, weights=weights)
        else:
            reward_calc = RewardFactory.create(self.config.environment.reward_version, board=board)

        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=RandomAgent(seed=self.config.seed),
            num_players=self.config.environment.players,
            reward_calculator=reward_calc,
        )

        evaluator = MultiOpponentEvaluator(
            board=board,
            tickets_deck=tickets,
            seed=self.config.seed + 1000,
        )
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)

        checkpoint_dir = self.config.training.checkpoint_dir
        exp_name = self.config.name
        latest_ckpt = os.path.join(checkpoint_dir, f"{exp_name}_latest.pt")
        best_ckpt = os.path.join(checkpoint_dir, f"{exp_name}_best.pt")

        best_win_rate = -1.0
        final_metrics: dict[str, float] = {}

        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dqn":
            dqn_config = {
                "lr": self.config.algorithm.learning_rate,
                "gamma": self.config.algorithm.gamma,
                "batch_size": self.config.algorithm.batch_size,
                "buffer_size": self.config.algorithm.buffer_size,
                "target_update_freq": self.config.algorithm.target_update_freq,
                "epsilon_start": self.config.algorithm.epsilon_start,
                "epsilon_end": self.config.algorithm.epsilon_end,
                "epsilon_decay_steps": self.config.algorithm.epsilon_decay_steps,
                "learning_starts": self.config.algorithm.learning_starts,
                "max_grad_norm": self.config.algorithm.max_grad_norm,
            }
            trainer = MaskedDQNTrainer(env=env, config=dqn_config)

            def eval_cb(step: int, tr: MaskedDQNTrainer) -> None:
                nonlocal best_win_rate, final_metrics
                eval_agent = DQNAgent(
                    input_dim=env.observation_space.shape[0],
                    action_dim=int(env.action_space.n),
                    encoder=env.encoder,
                    discrete_actions=env.discrete_actions,
                )
                eval_agent.q_net.load_state_dict(tr.policy_net.state_dict())
                eval_agent.q_net.eval()

                eval_results = evaluator.evaluate(
                    agent=eval_agent,
                    opponents=self.config.evaluation.opponents,
                    games_per_opponent=self.config.training.eval_episodes_per_opponent,
                )
                final_metrics.update(eval_results)
                tr.save(latest_ckpt)

                target_wr = eval_results.get("win_rate_vs_random", 0.0)
                if target_wr > best_win_rate:
                    best_win_rate = target_wr
                    tr.save(best_ckpt)

            trainer.train(
                total_timesteps=self.config.training.total_timesteps,
                eval_callback=eval_cb,
                eval_freq=self.config.training.eval_freq,
            )
            # Final evaluation
            eval_cb(trainer.total_timesteps, trainer)

        elif algo_name == "ppo":
            ppo_config = {
                "lr": self.config.algorithm.learning_rate,
                "gamma": self.config.algorithm.gamma,
                "gae_lambda": self.config.algorithm.gae_lambda,
                "clip_eps": self.config.algorithm.clip_range,
                "vf_coef": self.config.algorithm.value_coef,
                "ent_coef": self.config.algorithm.entropy_coef,
                "rollout_steps": self.config.training.rollout_steps,
                "num_epochs": self.config.training.num_epochs,
                "minibatch_size": self.config.training.batch_size,
                "max_grad_norm": self.config.algorithm.max_grad_norm,
            }
            trainer = MaskedPPOTrainer(env=env, config=ppo_config)

            def eval_cb_ppo(step: int, tr: MaskedPPOTrainer) -> None:
                nonlocal best_win_rate, final_metrics
                eval_agent = PPOAgent(
                    input_dim=env.observation_space.shape[0],
                    action_dim=int(env.action_space.n),
                    encoder=env.encoder,
                    discrete_actions=env.discrete_actions,
                )
                eval_agent.actor_critic.load_state_dict(tr.actor_critic.state_dict())
                eval_agent.actor_critic.eval()

                eval_results = evaluator.evaluate(
                    agent=eval_agent,
                    opponents=self.config.evaluation.opponents,
                    games_per_opponent=self.config.training.eval_episodes_per_opponent,
                )
                final_metrics.update(eval_results)
                tr.save(latest_ckpt)

                target_wr = eval_results.get("win_rate_vs_random", 0.0)
                if target_wr > best_win_rate:
                    best_win_rate = target_wr
                    tr.save(best_ckpt)

            trainer.train(
                total_timesteps=self.config.training.total_timesteps,
                eval_callback=eval_cb_ppo,
                eval_freq=self.config.training.eval_freq,
            )
            eval_cb_ppo(trainer.total_timesteps, trainer)

        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        record = ExperimentRecord(
            experiment_id=str(uuid.uuid4())[:8],
            name=self.config.name,
            seed=self.config.seed,
            algorithm=self.config.algorithm.name,
            env_version=self.config.environment.observation_version,
            reward_version=self.config.environment.reward_version,
            metrics=final_metrics,
        )
        self.registry.log_experiment(record)
        return record
