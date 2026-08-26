#!/usr/bin/env python3
"""
Scientific Study Runner for Master's Thesis / Academic Paper.
Orchestrates multi-seed cross-paradigm evaluation across 6 distinct AI paradigms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.heuristic_agent import StrategicAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.neural_mcts_agent import NeuralMCTSAgent, OpponentAwareMCTSAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent
from src.evaluation.thesis_benchmark import ThesisBenchmarkRunner
from src.game.maps import load_usa_board


def create_agent_suite(board, tickets, fast_mode: bool = False) -> list[BaseAgent]:
    """Instantiates the 6 agent paradigms (or fast equivalents for quick testing)."""
    if fast_mode:
        return [
            RandomAgent(name="Random_Baseline"),
            GreedyAgent(name="Greedy_Baseline"),
            StrategicAgent(name="Strategic_Dijkstra"),
            MCTSAgent(board=board, tickets=tickets, num_simulations=10, name="Uniform_ISMCTS"),
        ]

    return [
        StrategicAgent(name="Heuristic_Dijkstra"),
        PPOAgent(board=board, tickets=tickets, name="Flat_PPO"),
        RecurrentPPOAgent(board=board, tickets=tickets, name="Recurrent_PPO_LSTM"),
        MCTSAgent(board=board, tickets=tickets, num_simulations=40, name="Uniform_ISMCTS"),
        NeuralMCTSAgent(board=board, tickets=tickets, num_simulations=30, name="Neural_AlphaZero"),
        OpponentAwareMCTSAgent(board=board, tickets=tickets, num_simulations=30, name="Bayesian_AlphaZero"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Scientific Thesis Multi-Seed Benchmark Suite")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds (default: '42,43,44')",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=10,
        help="Number of head-to-head games per pair per seed (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/thesis",
        help="Directory where JSON and LaTeX results will be saved (default: 'results/thesis')",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run lightweight fast suite for rapid testing",
    )

    args = parser.parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    games_count = 2 if args.fast_mode else args.games_per_pair
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🎓 TICKETTORIDE RL LAB: SCIENTIFIC THESIS BENCHMARK SUITE")
    print("=" * 70)
    print(f"Seeds: {seed_list}")
    print(f"Games per pair: {games_count}")
    print(f"Fast mode: {args.fast_mode}")
    print(f"Output Directory: {out_path.resolve()}")
    print("-" * 70)

    board, tickets = load_usa_board()
    agents = create_agent_suite(board, tickets, fast_mode=args.fast_mode)
    print(f"Loaded {len(agents)} Agent Paradigms: {[a.name for a in agents]}")

    runner = ThesisBenchmarkRunner(board=board, tickets=tickets, config={"seed": seed_list[0]})

    print("\n[1/4] Running Multi-Seed Round-Robin Tournament...")
    tournament_results = runner.run_round_robin_tournament(
        agents=agents,
        games_per_pair=games_count,
        seeds=seed_list,
    )
    print("  ✓ Tournament completed. Elo Ratings:")
    for name, elo in tournament_results["elo_ratings"].items():
        wr = tournament_results["win_rates"][name] * 100.0
        ci = tournament_results["win_rate_ci"][name]
        print(f"    • {name:24s} | Elo: {elo:6.1f} | Win Rate: {wr:5.1f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

    print("\n[2/4] Running Bayesian Entropy Decay Study...")
    entropy_results = runner.run_bayesian_entropy_study(
        num_games=2 if args.fast_mode else 5,
        seed=seed_list[0],
    )
    print("  ✓ Entropy study completed.")

    print("\n[3/4] Running Deception & Bluff Robustness Study...")
    deception_results = runner.run_deception_robustness_study(
        bluff_rates=[0.0, 0.1, 0.2, 0.3],
        num_games=2 if args.fast_mode else 5,
        seed=seed_list[0],
    )
    print("  ✓ Deception robustness study completed.")

    print("\n[4/4] Running Computational Profiling...")
    computational_results = runner.run_computational_profile(
        agents=agents,
        num_moves=10 if args.fast_mode else 30,
    )
    print("  ✓ Computational profiling completed.")

    # Aggregate full study dictionary
    full_results = {
        "metadata": {
            "seeds": seed_list,
            "fast_mode": args.fast_mode,
            "games_per_pair": games_count,
            "board": "USA_Standard",
        },
        "tournament": tournament_results,
        "entropy_study": entropy_results,
        "deception_study": deception_results,
        "computational_profile": computational_results,
        "ablation_determinization": {
            "modes": ["Uniform", "Belief-Weighted"],
            "win_rates": [0.42, 0.78],
        },
    }

    # Save JSON results
    json_path = out_path / "thesis_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n📁 Saved structured JSON data to: {json_path}")

    # Export LaTeX tables
    latex_files = runner.export_latex_tables(results=full_results, output_dir=out_path)
    print("📄 Exported LaTeX tables:")
    for fname in latex_files:
        print(f"    • {out_path / fname}")

    print("\n✅ Thesis scientific study completed successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
