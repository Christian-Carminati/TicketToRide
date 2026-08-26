"""Unit tests for RewardResearchRunner and scientific report generation."""

from pathlib import Path

from src.evaluation.reward_research import RewardResearchRunner


def test_reward_research_study_quick_execution(tmp_path: Path):
    json_path = tmp_path / "study.json"
    md_path = tmp_path / "study.md"

    runner = RewardResearchRunner(
        board_type="mini",
        reward_versions=["v1", "v2", "v3", "v4"],
        training_steps=200,
        eval_games=2,
        output_json=str(json_path),
        output_md=str(md_path),
        seed=42,
    )

    results = runner.run_study()

    assert "reward_studies" in results
    assert "v1" in results["reward_studies"]
    assert "v2" in results["reward_studies"]
    assert "v3" in results["reward_studies"]
    assert "v4" in results["reward_studies"]

    assert json_path.exists()
    assert md_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "# Reward Research & Behavioral Benchmarking Report" in md_content
    assert "Ticket Completion" in md_content
    assert "Route Efficiency" in md_content
    assert "Reward V1 (Sparse / Pure Outcome)" in md_content
