"""Experiment Configuration schemas using Pydantic."""

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int | str = 1
    route_points_weight: float = 1.0
    ticket_completion_weight: float = 1.0
    step_penalty: float = 0.0
    win_bonus: float = 20.0
    loss_penalty: float = 10.0
    score_diff_weight: float = 0.5
    ticket_failure_penalty_weight: float = 1.0


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    board: str = "usa"
    players: int = 2
    observation_version: int = 1
    reward_version: int | str = 1
    reward_config: RewardConfig | None = None


class AlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "ppo"
    recurrent: bool = False
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    buffer_size: int = 50000
    batch_size: int = 64
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    learning_starts: int = 500
    max_grad_norm: float = 0.5


class NetworkConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hidden_dim: int = 128
    num_layers: int = 2
    lstm_hidden_dim: int | None = None


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_timesteps: int = 50000
    batch_size: int = 64
    rollout_steps: int = 512
    num_epochs: int = 4
    eval_freq: int = 5000
    eval_episodes_per_opponent: int = 20
    checkpoint_freq: int = 25000
    checkpoint_dir: str = "experiments/checkpoints"


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    opponents: list[str] = Field(default_factory=lambda: ["random", "greedy", "strategic"])
    num_episodes: int = 20


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    seed: int = 42
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
