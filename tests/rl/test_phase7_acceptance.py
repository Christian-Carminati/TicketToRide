"""Phase 7 Acceptance Test Suite: Reward Research & Behavioral Benchmarking."""

from pathlib import Path
import pytest

from src.environment.reward import (
    CustomRewardCalculator,
    RewardFactory,
    RewardV1_Sparse,
    RewardV2_DenseRoutes,
    RewardV3_TicketMilestones,
    RewardV4_StrategicShaped,
    RewardWeights,
)
from src.evaluation.reward_research import RewardResearchRunner


def test_phase7_reward_hierarchy_and_factories():
    """Acceptance Criterion 1: All reward versions exist, register aliases, and provide telemetry breakdowns."""
    for v in ["v1", "v2", "v3", "v4"]:
        calc = RewardFactory.create(v)
        assert calc is not None

    r1 = RewardFactory.create("sparse")
    assert isinstance(r1, RewardV1_Sparse)

    r2 = RewardFactory.create("dense_routes")
    assert isinstance(r2, RewardV2_DenseRoutes)

    r3 = RewardFactory.create("ticket_milestones")
    assert isinstance(r3, RewardV3_TicketMilestones)

    r4 = RewardFactory.create("strategic")
    assert isinstance(r4, RewardV4_StrategicShaped)

    rc = RewardFactory.create("custom", weights=RewardWeights(win_bonus=100.0))
    assert isinstance(rc, CustomRewardCalculator)


def test_phase7_automated_reward_comparison_study(tmp_path: Path):
    """Acceptance Criterion 2: Automated study trains multi-reward agents and measures behavioral divergence."""
    json_path = tmp_path / "acceptance_report.json"
    md_path = tmp_path / "acceptance_report.md"

    runner = RewardResearchRunner(
        board_type="mini",
        reward_versions=["v1", "v2", "v3", "v4"],
        training_steps=400,
        eval_games=4,
        output_json=str(json_path),
        output_md=str(md_path),
        seed=123,
    )
    results = runner.run_study()

    assert len(results["reward_studies"]) == 4
    assert json_path.exists()
    assert md_path.exists()

    # Verify structured metrics exist for all versions
    for v_key, study in results["reward_studies"].items():
        assert "vs_random" in study
        assert "ticket_completion_rate" in study["vs_random"]
        assert "route_efficiency" in study["vs_random"]
        assert "avg_routes_claimed" in study["vs_random"]
        assert "avg_game_turns" in study["vs_random"]
        assert "cards_drawn_ratio" in study["vs_random"]

    md_text = md_path.read_text(encoding="utf-8")
    assert "# Reward Research & Behavioral Benchmarking Report" in md_text
    assert "V1 (Sparse Outcome)" in md_text
    assert "V2 (Dense Routes)" in md_text
    assert "V3 (Ticket Milestones)" in md_text
    assert "V4 (Strategic Shaped)" in md_text
