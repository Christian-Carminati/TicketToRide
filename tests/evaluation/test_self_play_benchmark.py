from pathlib import Path
import json
import pytest
from src.evaluation.self_play_benchmark import SelfPlayBenchmarkRunner


def test_self_play_benchmark_study_and_report(tmp_path: Path):
    runner = SelfPlayBenchmarkRunner(
        config={
            "seed": 42,
            "training_steps": 128,
            "snapshot_interval": 64,
            "eval_games": 2,
            "games_per_pair": 2,
        }
    )
    results = runner.run_study()
    assert "tournament" in results
    assert "leaderboard" in results["tournament"]
    assert "vs_baselines" in results
    assert "metadata" in results

    md_path = str(tmp_path / "phase9_report.md")
    json_path = str(tmp_path / "phase9_report.json")
    runner.generate_report(results, output_md_path=md_path, output_json_path=json_path)

    assert Path(md_path).exists()
    assert Path(json_path).exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "tournament" in data
    assert "vs_baselines" in data
