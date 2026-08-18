"""Headless training entrypoint."""

import argparse

from src.experiments.config import ExperimentConfig
from src.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent headless")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to experiment YAML configuration file",
    )
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    config = ExperimentConfig.from_yaml(args.config)
    print(f"Starting experiment: {config.name} (Seed: {config.seed}, Algorithm: {config.algorithm.name})")

    runner = ExperimentRunner(config)
    record = runner.run()
    print(f"Experiment finished. ID: {record.experiment_id}")


if __name__ == "__main__":
    main()
