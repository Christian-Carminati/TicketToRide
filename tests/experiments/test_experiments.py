"""Unit tests for Experiment configs and registry."""

import os
import tempfile

from src.experiments.config import ExperimentConfig
from src.experiments.registry import ExperimentRegistry
from src.experiments.runner import ExperimentRunner


def test_experiment_config_roundtrip():
    config = ExperimentConfig(name="test_exp", seed=123)
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name

    try:
        config.to_yaml(path)
        loaded = ExperimentConfig.from_yaml(path)
        assert loaded.name == "test_exp"
        assert loaded.seed == 123
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_experiment_registry_and_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_file = os.path.join(tmpdir, "registry.jsonl")
        config = ExperimentConfig(name="exp_run_test")
        runner = ExperimentRunner(config)
        runner.registry = ExperimentRegistry(registry_file=registry_file)

        record = runner.run()
        assert record.name == "exp_run_test"
        assert os.path.exists(registry_file)

        records = runner.registry.list_experiments()
        assert len(records) == 1
        assert records[0]["name"] == "exp_run_test"
