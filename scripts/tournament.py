"""Headless tournament entrypoint."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.tournament import Tournament


def parse_agent_list(agents_str: str, seed: int) -> list[BaseAgent]:
    tokens = [t.strip().lower() for t in agents_str.split(",") if t.strip()]
    agents: list[BaseAgent] = []
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
        name = f"{t.capitalize()}_{counts[t]}"
        if t == "random":
            agents.append(RandomAgent(seed=seed + len(agents), name=name))
        elif t in ["greedy", "heuristic"]:
            agents.append(GreedyAgent(name=name))
        elif t == "strategic":
            agents.append(StrategicHeuristicAgent(name=name))
        else:
            raise ValueError(f"Unknown agent type: {t}")
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a round-robin tournament among agents"
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="random,greedy,strategic",
        help="Comma-separated agent types",
    )
    parser.add_argument(
        "--games-per-pair", type=int, default=50, help="Games per pair of agents"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Tournament seed"
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional path to export JSON tournament report",
    )
    args = parser.parse_args()

    agents = parse_agent_list(args.agents, seed=args.seed)
    tournament = Tournament(agents=agents, games_per_pair=args.games_per_pair)
    results = tournament.run(seed=args.seed)

    print("\n" + "=" * 75)
    print(
        f"  Tournament Leaderboard ({len(agents)} agents | {args.games_per_pair} games/matchup | Seed: {args.seed})"
    )
    print("=" * 75)
    print(
        f"{'Rank':<5} | {'Agent':<18} | {'Elo':<7} | {'Win Rate':<10} | {'W-L-D':<12} | {'Avg Score':<10}"
    )
    print("-" * 75)
    for rank, entry in enumerate(results["leaderboard"], 1):
        wld = f"{entry['wins']}-{entry['losses']}-{entry['draws']}"
        print(
            f"{rank:<5} | {entry['name']:<18} | {entry['elo']:<7.1f} | {entry['win_rate'] * 100:<9.1f}% | "
            f"{wld:<12} | {entry['avg_score']:<10.1f}"
        )
    print("=" * 75 + "\n")

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Tournament report saved to {args.export_json}")


if __name__ == "__main__":
    main()
