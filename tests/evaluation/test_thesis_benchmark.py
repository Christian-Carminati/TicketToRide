"""Tests for the Thesis Scientific Benchmark Runner and LaTeX Table Exporter."""

import pytest
import numpy as np
from pathlib import Path
from src.game.maps import load_usa_board
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.evaluation.thesis_benchmark import ThesisBenchmarkRunner


def test_thesis_benchmark_runner_init():
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets, config={"seed": 42})
    assert runner.seed == 42
    assert runner.board == board
    assert len(runner.tickets) == len(tickets)


def test_thesis_benchmark_round_robin_and_latex(tmp_path: Path):
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets)
    
    agent1 = RandomAgent(name="Random_1")
    agent2 = GreedyAgent(name="Greedy_2")
    
    res = runner.run_round_robin_tournament(agents=[agent1, agent2], games_per_pair=2, seeds=[42])
    assert "payoff_matrix" in res
    assert "elo_ratings" in res
    assert "win_rates" in res
    assert "Random_1" in res["elo_ratings"]
    assert "Greedy_2" in res["elo_ratings"]
    assert "Random_1" in res["win_rates"]
    
    latex_files = runner.export_latex_tables(results={"tournament": res}, output_dir=tmp_path)
    assert "table1_main_results.tex" in latex_files
    assert (tmp_path / "table1_main_results.tex").exists()
    content = (tmp_path / "table1_main_results.tex").read_text()
    assert "\\begin{table}" in content


def test_thesis_benchmark_entropy_and_deception():
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets)
    
    entropy_res = runner.run_bayesian_entropy_study(num_games=2, seed=42)
    assert "turns" in entropy_res
    assert "mean_entropy" in entropy_res
    assert "top1_accuracy" in entropy_res
    assert len(entropy_res["turns"]) > 0
    assert len(entropy_res["mean_entropy"]) == len(entropy_res["turns"])
    
    deception_res = runner.run_deception_robustness_study(bluff_rates=[0.0, 0.2], num_games=2, seed=42)
    assert "bluff_rates" in deception_res
    assert 0.0 in deception_res["bluff_rates"]
    assert 0.2 in deception_res["bluff_rates"]
    assert "bayesian_elo" in deception_res
    assert "lstm_elo" in deception_res


def test_thesis_benchmark_computational_profile(tmp_path: Path):
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets)
    
    agent1 = RandomAgent(name="Random_1")
    agent2 = GreedyAgent(name="Greedy_2")
    
    prof_res = runner.run_computational_profile(agents=[agent1, agent2], num_moves=10)
    assert "agents" in prof_res
    assert "ms_per_move" in prof_res
    assert len(prof_res["agents"]) == 2
    
    latex_files = runner.export_latex_tables(
        results={"computational_profile": prof_res},
        output_dir=tmp_path
    )
    assert "table3_computational_profile.tex" in latex_files
    assert (tmp_path / "table3_computational_profile.tex").exists()
