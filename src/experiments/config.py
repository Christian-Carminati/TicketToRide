"""Experiment Configuration schemas using Pydantic."""

import yaml
from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    players: int = 2
    observation_version: int = 1
    reward_version: int = 1


class AlgorithmConfig(BaseModel):
    name: str = "ppo"
    recurrent: bool = False
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5


class NetworkConfig(BaseModel):
    hidden_dim: int = 128
    num_layers: int = 2
    lstm_hidden_dim: int | None = None


class TrainingConfig(BaseModel):
    total_timesteps: int = 100_000
    batch_size: int = 64
    rollout_steps: int = 2048
    eval_freq: int = 10_000
    checkpoint_freq: int = 25_000


class EvaluationConfig(BaseModel):
    opponents: list[str] = Field(default_factory=lambda: ["random", "greedy"])
    num_episodes: int = 100


class ExperimentConfig(BaseModel):
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
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
