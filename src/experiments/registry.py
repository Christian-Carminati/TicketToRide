"""Experiment tracking and metadata registry."""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentRecord:
    experiment_id: str
    name: str
    seed: int
    algorithm: str
    env_version: int
    reward_version: int
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    created_at: str | None = None


class ExperimentRegistry:
    """Stores and retrieves experiment records from local SQLite or JSON."""

    def __init__(self, registry_file: str = "experiments/results/registry.jsonl") -> None:
        self.registry_file = registry_file
        os.makedirs(os.path.dirname(registry_file), exist_ok=True)

    def log_experiment(self, record: ExperimentRecord) -> None:
        with open(self.registry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def list_experiments(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.registry_file):
            return []
        records = []
        with open(self.registry_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records
