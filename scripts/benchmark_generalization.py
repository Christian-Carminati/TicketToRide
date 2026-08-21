"""CLI Benchmark Runner for Phase 10: Generalization & Procedural Maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.generalization import GeneralizationBenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket to Ride RL Lab — Phase 10 Generalization Benchmark")
    parser.add_argument("--num-train-maps", type=int, default=5, help="Number of procedural train maps")
    parser.add_argument("--num-test-maps", type=int, default=3, help="Number of unseen procedural test maps")
    parser.add_argument("--training-steps", type=int, default=2000, help="Timesteps to train RL models")
    parser.add_argument("--games-per-map", type=int, default=5, help="Evaluation games per map")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument(
        "--report-md",
        type=str,
        default="experiments/results/phase10_report.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="experiments/results/phase10_report.json",
        help="JSON report path",
    )

    args = parser.parse_args()

    print("================================================================")
    print(" Ticket to Ride RL Lab — Phase 10 Generalization Benchmark Study")
    print("================================================================")
    print(f" Train Maps:     {args.num_train_maps}")
    print(f" Test Maps:      {args.num_test_maps} (unseen)")
    print(f" Training Steps: {args.training_steps}")
    print(f" Games / Map:    {args.games_per_map}")
    print(f" Random Seed:    {args.seed}")
    print("----------------------------------------------------------------\n")

    config = {
        "train_seeds": list(range(1, args.num_train_maps + 1)),
        "test_seeds": list(range(101, 101 + args.num_test_maps)),
        "training_steps": args.training_steps,
        "games_per_map": args.games_per_map,
        "seed": args.seed,
    }

    runner = GeneralizationBenchmarkRunner(config=config)
    results = runner.run_study()
    runner.generate_report(results, output_md_path=args.report_md, output_json_path=args.report_json)

    print(f"\n[SUCCESS] Benchmark study complete. Reports saved to:")
    print(f" -> {args.report_md}")
    print(f" -> {args.report_json}")


if __name__ == "__main__":
    main()
