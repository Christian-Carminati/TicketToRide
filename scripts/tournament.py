"""Headless tournament entrypoint."""

import argparse

from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.tournament import Tournament


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a round-robin tournament")
    parser.add_argument("--games-per-pair", type=int, default=50, help="Games per pair of agents")
    parser.add_argument("--seed", type=int, default=42, help="Tournament seed")
    args = parser.parse_args()

    agents = [
        RandomAgent(seed=args.seed, name="Random_1"),
        RandomAgent(seed=args.seed + 1, name="Random_2"),
        HeuristicAgent(name="Greedy_1"),
    ]

    tournament = Tournament(agents=agents, games_per_pair=args.games_per_pair)
    results = tournament.run(seed=args.seed)
    print("Tournament Results:")
    for agent_name in results:
        rating = tournament.elo_system.get_rating(agent_name)
        print(f"- {agent_name}: Elo {rating:.1f}")


if __name__ == "__main__":
    main()
