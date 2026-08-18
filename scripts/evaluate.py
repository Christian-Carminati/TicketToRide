"""Headless Agent Evaluation entrypoint supporting Heuristics and trained RL models."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import ObservationV1
from src.evaluation.evaluator import Evaluator
from src.game.maps import create_synthetic_mini_board, load_usa_board


def build_agent(
    agent_type: str,
    seed: int = 42,
    name: str | None = None,
    checkpoint: str | None = None,
    board=None,
    tickets=None,
) -> BaseAgent:
    name_str = name or agent_type.capitalize()
    t = agent_type.lower()
    if t == "random":
        return RandomAgent(seed=seed, name=name_str)
    elif t in ["greedy", "heuristic"]:
        return GreedyAgent(name=name_str)
    elif t == "strategic":
        return StrategicHeuristicAgent(name=name_str)
    elif t == "dqn":
        encoder = ObservationV1(board=board, initial_tickets=tickets or []) if board else None
        discrete_actions = DiscreteActionSpace(board=board) if board else None
        obs_dim = encoder.encode(None).shape[0] if encoder else 100
        action_dim = discrete_actions.size if discrete_actions else 56
        return DQNAgent(
            name=name_str,
            model_path=checkpoint,
            input_dim=obs_dim,
            action_dim=action_dim,
            encoder=encoder,
            discrete_actions=discrete_actions,
        )
    elif t == "ppo":
        encoder = ObservationV1(board=board, initial_tickets=tickets or []) if board else None
        discrete_actions = DiscreteActionSpace(board=board) if board else None
        obs_dim = encoder.encode(None).shape[0] if encoder else 100
        action_dim = discrete_actions.size if discrete_actions else 56
        return PPOAgent(
            name=name_str,
            model_path=checkpoint,
            input_dim=obs_dim,
            action_dim=action_dim,
            encoder=encoder,
            discrete_actions=discrete_actions,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two agents head-to-head")
    parser.add_argument(
        "--agent1",
        type=str,
        default="strategic",
        help="Type of Agent 1 (random, greedy, strategic, dqn, ppo)",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default="greedy",
        help="Type of Agent 2 (random, greedy, strategic, dqn, ppo)",
    )
    parser.add_argument(
        "--checkpoint1",
        type=str,
        default=None,
        help="Path to .pt checkpoint for Agent 1 (if RL agent)",
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        default=None,
        help="Path to .pt checkpoint for Agent 2 (if RL agent)",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="usa",
        choices=["mini", "usa"],
        help="Board type (mini or usa)",
    )
    parser.add_argument(
        "--games", type=int, default=100, help="Number of games to simulate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic evaluation seed"
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional path to export JSON metrics",
    )
    args = parser.parse_args()

    board, tickets = create_synthetic_mini_board() if args.board == "mini" else load_usa_board()

    a1 = build_agent(
        args.agent1,
        seed=args.seed,
        name=f"{args.agent1.capitalize()}_1",
        checkpoint=args.checkpoint1,
        board=board,
        tickets=tickets,
    )
    a2 = build_agent(
        args.agent2,
        seed=args.seed + 1,
        name=f"{args.agent2.capitalize()}_2",
        checkpoint=args.checkpoint2,
        board=board,
        tickets=tickets,
    )

    evaluator = Evaluator(board=board, tickets_deck=tickets, seed=args.seed)
    results = evaluator.evaluate(agent_a=a1, agent_b=a2, num_games=args.games, seed=args.seed)

    m1 = results[a1.name]
    m2 = results[a2.name]

    print("=" * 65)
    print(f"  Head-to-Head Evaluation: {a1.name} vs {a2.name}")
    print(f"  Total Games: {args.games} | Seed: {args.seed} | Board: {args.board}")
    print("=" * 65)
    print(f"{'Metric':<25} | {a1.name:<16} | {a2.name:<16}")
    print("-" * 65)
    print(f"{'Wins':<25} | {m1.wins:<16} | {m2.wins:<16}")
    print(
        f"{'Win Rate':<25} | {m1.win_rate * 100:<15.1f}% | {m2.win_rate * 100:<15.1f}%"
    )
    print(f"{'Draws':<25} | {m1.draws:<16} | {m2.draws:<16}")
    print(f"{'Avg Score':<25} | {m1.avg_score:<16.1f} | {m2.avg_score:<16.1f}")
    print(
        f"{'Avg Score Diff':<25} | {m1.avg_score_diff:<+16.1f} | {m2.avg_score_diff:<+16.1f}"
    )
    print(
        f"{'Ticket Completion Rate':<25} | {m1.ticket_completion_rate * 100:<15.1f}% | {m2.ticket_completion_rate * 100:<15.1f}%"
    )
    print(f"{'Avg Turns / Game':<25} | {m1.avg_turns:<16.1f} | {m2.avg_turns:<16.1f}")
    print("=" * 65)

    if args.export_json:
        data = {
            a1.name: {
                "wins": m1.wins,
                "win_rate": m1.win_rate,
                "avg_score": m1.avg_score,
                "ticket_rate": m1.ticket_completion_rate,
            },
            a2.name: {
                "wins": m2.wins,
                "win_rate": m2.win_rate,
                "avg_score": m2.avg_score,
                "ticket_rate": m2.ticket_completion_rate,
            },
        }
        with open(args.export_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results exported to {args.export_json}")


if __name__ == "__main__":
    main()
