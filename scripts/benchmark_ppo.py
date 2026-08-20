#!/usr/bin/env python3
"""CLI script to run PPO benchmarks and ablation studies."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import PPOBenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO Benchmark and Ablation Study.")
    parser.add_argument("--board", choices=["mini", "usa"], default="mini", help="Board type (mini or usa)")
    parser.add_argument("--games", type=int, default=30, help="Games per opponent in evaluation")
    parser.add_argument("--steps", type=int, default=2000, help="Total training timesteps for primary PPO agent")
    parser.add_argument("--ablation", action="store_true", default=True, help="Run CleanRL ablation study")
    parser.add_argument("--ablation-steps", type=int, default=500, help="Training timesteps per ablation variant")
    parser.add_argument("--ablation-games", type=int, default=15, help="Games per opponent in ablation study")
    parser.add_argument("--output-json", default="experiments/results/benchmark_phase6.json", help="Path to output JSON")
    parser.add_argument("--output-md", default="experiments/results/benchmark_phase6.md", help="Path to output Markdown")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print(f"🚀 Starting PPO Benchmark on {args.board.upper()} map ({args.steps} training steps)...")
    runner = PPOBenchmarkRunner(
        board_type=args.board,
        games_per_opponent=args.games,
        total_training_steps=args.steps,
        output_json=args.output_json,
        output_md=args.output_md,
        seed=args.seed,
    )

    results = runner.run_benchmark(
        run_ablation=args.ablation,
        ablation_steps=args.ablation_steps,
        ablation_games=args.ablation_games,
    )

    print("\n✅ Benchmark Complete!")
    print(f"📄 JSON Report saved to: {args.output_json}")
    print(f"📄 Markdown Report saved to: {args.output_md}")
    print("\n" + runner.generate_markdown_report(results))


if __name__ == "__main__":
    main()
