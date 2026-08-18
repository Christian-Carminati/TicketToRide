"""Experiment execution engine."""

import uuid

from src.experiments.config import ExperimentConfig
from src.experiments.registry import ExperimentRecord, ExperimentRegistry


class ExperimentRunner:
    """Orchestrates configuration-driven training, checkpointing, and evaluation."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.registry = ExperimentRegistry()

    def run(self) -> ExperimentRecord:
        # Skeleton experiment runner
        record = ExperimentRecord(
            experiment_id=str(uuid.uuid4())[:8],
            name=self.config.name,
            seed=self.config.seed,
            algorithm=self.config.algorithm.name,
            env_version=self.config.environment.observation_version,
            reward_version=self.config.environment.reward_version,
            metrics={"status": "completed"},
        )
        self.registry.log_experiment(record)
        return record
