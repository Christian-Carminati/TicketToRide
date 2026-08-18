"""Headless evaluation entrypoint."""

import argparse

from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent against baselines")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    args = parser.parse_args()

    agent = HeuristicAgent(name="GreedyAgent")
    opponent = RandomAgent(name="RandomAgent", seed=args.seed)

    evaluator = Evaluator()
    metrics = evaluator.evaluate(agent, opponent, num_episodes=args.episodes, seed=args.seed)
    print(f"Evaluation finished over {metrics.total_games} episodes.")
    print(f"Win rate: {metrics.win_rate:.2%}")


if __name__ == "__main__":
    main()
