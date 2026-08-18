"""Unit tests for Evaluation, Tournaments, and Elo system."""

from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.elo import EloSystem
from src.evaluation.evaluator import Evaluator
from src.evaluation.tournament import Tournament


def test_elo_update():
    elo = EloSystem(initial_rating=1200.0, k_factor=32.0)
    assert elo.get_rating("player1") == 1200.0

    # Player 1 wins
    elo.update("player1", "player2", score_a=1.0)
    assert elo.get_rating("player1") > 1200.0
    assert elo.get_rating("player2") < 1200.0


def test_evaluator_skeleton():
    evaluator = Evaluator()
    agent = HeuristicAgent()
    opponent = RandomAgent()
    metrics = evaluator.evaluate(agent, opponent, num_episodes=5)
    assert metrics.total_games == 5


def test_tournament_skeleton():
    agents = [RandomAgent(name="A1"), RandomAgent(name="A2")]
    tournament = Tournament(agents=agents, games_per_pair=2)
    results = tournament.run(seed=42)
    assert "A1" in results
    assert "A2" in results
