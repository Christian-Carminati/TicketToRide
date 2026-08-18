"""Unit tests for Evaluator head-to-head match runner."""

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.evaluator import Evaluator


def test_evaluator_head_to_head_alternates_first_player():
    evaluator = Evaluator()
    agent_a = RandomAgent(seed=10, name="Random_A")
    agent_b = RandomAgent(seed=20, name="Random_B")

    results = evaluator.evaluate(agent_a, agent_b, num_games=10, seed=42)

    assert "Random_A" in results
    assert "Random_B" in results
    metrics_a = results["Random_A"]
    metrics_b = results["Random_B"]

    assert metrics_a.total_games == 10
    assert metrics_b.total_games == 10
    assert metrics_a.wins + metrics_b.wins + metrics_a.draws == 10
    assert metrics_a.avg_turns > 0


def test_evaluator_greedy_beats_random():
    evaluator = Evaluator()
    greedy = GreedyAgent(name="Greedy")
    random_bot = RandomAgent(seed=100, name="Random")

    results = evaluator.evaluate(greedy, random_bot, num_games=20, seed=42)
    metrics_greedy = results["Greedy"]
    metrics_random = results["Random"]

    assert metrics_greedy.wins > metrics_random.wins
    assert metrics_greedy.avg_score > metrics_random.avg_score


def test_evaluator_strategic_beats_greedy():
    evaluator = Evaluator()
    strategic = StrategicHeuristicAgent(name="Strategic")
    greedy = GreedyAgent(name="Greedy")

    results = evaluator.evaluate(strategic, greedy, num_games=20, seed=42)
    metrics_strat = results["Strategic"]
    metrics_greedy = results["Greedy"]

    assert metrics_strat.wins > metrics_greedy.wins
    assert metrics_strat.avg_score > metrics_greedy.avg_score
