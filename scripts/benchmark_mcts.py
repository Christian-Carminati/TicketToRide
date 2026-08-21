#!/usr/bin/env python3
"""CLI tool for running Phase 11 MCTS Benchmarks."""

import argparse
from pathlib import Path

from src.evaluation.mcts_benchmark import MCTSBenchmarkRunner
from src.rl.mcts import MCTSConfig, RolloutPolicyType


def main() -> None:
    parser = argparse.ArgumentParser(description="TicketToRide Phase 11 MCTS Benchmark Suite")
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="Number of MCTS simulations per turn",
    )
    parser.add_argument("--depth", type=int, default=8, help="Max rollout depth")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "greedy", "strategic"],
        default="strategic",
        help="Rollout policy",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games per matchup",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output",
        type=str,
        default="phase11_report.md",
        help="Markdown output path",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="phase11_report.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    print(
        f"=== Running Phase 11 MCTS Benchmark (N={args.simulations}, policy={args.policy}, games={args.games}) ==="
    )
    config = MCTSConfig(
        num_simulations=args.simulations,
        max_rollout_depth=args.depth,
        rollout_policy=RolloutPolicyType(args.policy),
        seed=args.seed,
    )
    runner = MCTSBenchmarkRunner(config)
    results = runner.run_head_to_head_suite(num_games=args.games, seed=args.seed)

    runner.generate_report(results, output_md_path=args.output, output_json_path=args.json)
    print(f"Benchmark completed successfully! Report generated at: {args.output}")


if __name__ == "__main__":
    main()
