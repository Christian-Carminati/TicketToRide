"""Unit tests for MCTS benchmark runner."""

from pathlib import Path
from src.evaluation.mcts_benchmark import MCTSBenchmarkRunner
from src.rl.mcts import MCTSConfig, RolloutPolicyType


def test_mcts_benchmark_runner_quick(tmp_path: Path):
    """Verify MCTSBenchmarkRunner runs head-to-head matches and produces valid summaries and reports."""
    config = MCTSConfig(
        num_simulations=10,
        max_rollout_depth=3,
        rollout_policy=RolloutPolicyType.RANDOM,
        seed=42,
    )
    runner = MCTSBenchmarkRunner(mcts_config=config)

    results = runner.run_head_to_head_suite(num_games=2, seed=42)

    assert "opponents" in results
    assert "RandomAgent" in results["opponents"]
    assert "GreedyAgent" in results["opponents"]
    assert "StrategicAgent" in results["opponents"]
    assert "summary" in results
    assert "avg_win_rate" in results["summary"]

    md_out = tmp_path / "report.md"
    json_out = tmp_path / "report.json"
    runner.generate_report(results, output_md_path=md_out, output_json_path=json_out)

    assert md_out.exists()
    assert json_out.exists()
    assert len(md_out.read_text(encoding="utf-8")) > 50
