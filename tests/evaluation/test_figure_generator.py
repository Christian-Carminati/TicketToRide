"""Tests for the Thesis Publication Figure Generator."""

import json
from pathlib import Path
import pytest

from scripts.generate_thesis_figures import generate_all_figures


def test_generate_all_figures(tmp_path: Path):
    mock_data = {
        "tournament": {
            "agents": ["Heuristic", "Recurrent_PPO", "Bayesian_MCTS"],
            "elo_ratings": {"Heuristic": 1200.0, "Recurrent_PPO": 1460.0, "Bayesian_MCTS": 1550.0},
            "elo_ci": {
                "Heuristic": [1150.0, 1250.0],
                "Recurrent_PPO": [1420.0, 1500.0],
                "Bayesian_MCTS": [1500.0, 1600.0],
            },
            "payoff_matrix": [
                [0.5, 0.2, 0.1],
                [0.8, 0.5, 0.35],
                [0.9, 0.65, 0.5],
            ],
            "win_rates": {"Heuristic": 0.15, "Recurrent_PPO": 0.55, "Bayesian_MCTS": 0.80},
        },
        "entropy_study": {
            "turns": [1, 5, 10, 15, 20, 25],
            "mean_entropy": [4.5, 3.8, 2.8, 1.9, 1.1, 0.5],
            "top1_accuracy": [0.1, 0.25, 0.45, 0.65, 0.82, 0.94],
            "top3_accuracy": [0.3, 0.50, 0.70, 0.85, 0.95, 0.99],
        },
        "ablation_determinization": {
            "modes": ["Uniform", "Belief-Weighted"],
            "win_rates": [0.42, 0.78],
        },
        "deception_study": {
            "bluff_rates": [0.0, 0.1, 0.2, 0.3],
            "bayesian_elo": [1550.0, 1510.0, 1420.0, 1340.0],
            "lstm_elo": [1460.0, 1450.0, 1430.0, 1410.0],
        },
        "computational_profile": {
            "agents": ["Flat_PPO", "Recurrent_PPO", "Uniform_MCTS", "Bayesian_MCTS"],
            "ms_per_move": [0.8, 2.1, 45.0, 52.0],
            "elo": [1320.0, 1460.0, 1410.0, 1550.0],
        },
    }

    json_path = tmp_path / "mock_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    out_dir = tmp_path / "figures"
    generated = generate_all_figures(results_path=json_path, output_dir=out_dir)

    assert len(generated) >= 5
    for fig_path in generated:
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0
