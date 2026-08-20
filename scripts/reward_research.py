"""CLI Script for running automated Reward Research and Behavioral Benchmarking."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.reward_research import RewardResearchRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run automated multi-reward comparative research study for Ticket to Ride RL."
    )
    parser.add_argument(
        "--board",
        type=str,
        default="mini",
        choices=["mini", "usa"],
        help="Board map to run study on (default: mini)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=4000,
        help="Training timesteps per reward version (default: 4000)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=20,
        help="Evaluation games per baseline opponent (default: 20)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/results/reward_research.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="experiments/results/reward_research.md",
        help="Output path for Markdown report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 TICKET TO RIDE RL LAB — REWARD RESEARCH STUDY")
    print("=" * 60)
    print(f"Map: {args.board.upper()}")
    print(f"Training steps per version: {args.timesteps}")
    print(f"Evaluation games per opponent: {args.eval_games}")
    print(f"Seed: {args.seed}")
    print("Running multi-reward comparative study (v1, v2, v3, v4)...")

    runner = RewardResearchRunner(
        board_type=args.board,
        reward_versions=["v1", "v2", "v3", "v4"],
        training_steps=args.timesteps,
        eval_games=args.eval_games,
        output_json=args.output_json,
        output_md=args.output_md,
        seed=args.seed,
    )

    results = runner.run_study()

    print("\n✅ Study completed successfully!")
    print(f"JSON report written to: {args.output_json}")
    print(f"Markdown report written to: {args.output_md}")
    print("=" * 60)


if __name__ == "__main__":
    main()
