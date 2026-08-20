import json
from pathlib import Path

import pytest

from src.evaluation.benchmark import PPOBenchmarkRunner


def test_benchmark_runner_quick_evaluation(tmp_path: Path) -> None:
    json_path = tmp_path / "benchmark_test.json"
    md_path = tmp_path / "benchmark_test.md"

    runner = PPOBenchmarkRunner(
        board_type="mini",
        games_per_opponent=4,
        training_steps=300,
        output_json=str(json_path),
        output_md=str(md_path),
        seed=42,
    )

    results = runner.run_benchmark(run_ablation=True, ablation_steps=200, ablation_games=2)

    assert "opponents" in results
    assert "ablation" in results
    assert "random" in results["opponents"]
    assert "greedy" in results["opponents"]
    assert "strategic" in results["opponents"]

    # Verify JSON file generated and valid
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["board_type"] == "mini"
    assert "opponents" in data

    # Verify Markdown file generated and formatted
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "# PPO Benchmark & Ablation Report" in content
    assert "Ablation Study" in content
