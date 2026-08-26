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
    env_version: int | str
    reward_version: int | str
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    created_at: str | None = None


class ExperimentRegistry:
    """Stores and retrieves experiment records from local JSONL file."""

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
        with open(self.registry_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return records

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete a single experiment by ID."""
        if not os.path.exists(self.registry_file):
            return False

        records = self.list_experiments()
        initial_len = len(records)
        filtered = [r for r in records if r.get("experiment_id") != experiment_id]

        if len(filtered) == initial_len:
            return False

        with open(self.registry_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in filtered)
        return True

    def clear_all_experiments(self) -> int:
        """Delete all experiments in the registry."""
        if not os.path.exists(self.registry_file):
            return 0

        count = len(self.list_experiments())
        with open(self.registry_file, "w", encoding="utf-8") as f:
            f.truncate(0)
        return count
