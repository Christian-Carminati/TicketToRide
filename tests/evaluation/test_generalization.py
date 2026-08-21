"""Tests for GeneralizationEvaluator and GeneralizationBenchmarkRunner."""

import pytest
from src.agents.greedy_agent import GreedyAgent
from src.agents.heuristic_agent import StrategicAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.generalization import (
    GeneralizationBenchmarkRunner,
    GeneralizationEvaluator,
    GeneralizationResult,
)
from src.game.procedural import ProceduralMapDataset, ProceduralMapGenerator


def test_generalization_evaluator_metrics():
    gen = ProceduralMapGenerator()
    dataset = ProceduralMapDataset(gen)
    split = dataset.create_split(train_seeds=[1, 2], val_seeds=[10], test_seeds=[100, 101])

    evaluator = GeneralizationEvaluator(map_split=split, games_per_map=2, seed=42)
    agent_a = StrategicAgent(name="Strategic")
    agent_b = RandomAgent(name="Random", seed=42)

    result = evaluator.evaluate_agent_generalization(agent=agent_a, opponent=agent_b)

    assert isinstance(result, GeneralizationResult)
    assert result.train_map_score > 0
    assert result.unseen_map_score > 0
    assert isinstance(result.generalization_gap, float)
    assert isinstance(result.retention_rate, float)
    assert result.unseen_win_rate >= 0.0


def test_generalization_benchmark_study(tmp_path):
    runner = GeneralizationBenchmarkRunner(
        config={
            "train_seeds": [1, 2],
            "test_seeds": [100, 101],
            "games_per_map": 2,
            "training_steps": 500,
            "seed": 42,
        }
    )
    study_results = runner.run_study()

    assert "metadata" in study_results
    assert "heuristic_generalization" in study_results
    assert "rl_generalization" in study_results
    assert "official_cross_map" in study_results

    # Test report generation
    md_file = str(tmp_path / "test_report.md")
    json_file = str(tmp_path / "test_report.json")
    runner.generate_report(study_results, output_md_path=md_file, output_json_path=json_file)

    assert (tmp_path / "test_report.md").exists()
    assert (tmp_path / "test_report.json").exists()
