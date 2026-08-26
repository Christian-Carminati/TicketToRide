"""Phase 11 Acceptance Test Suite for MCTS & Heuristic Rollout."""

from pathlib import Path

from src.agents.greedy_agent import GreedyAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.evaluation.evaluator import Evaluator
from src.evaluation.mcts_benchmark import MCTSBenchmarkRunner
from src.rl.mcts import MCTSConfig, RolloutPolicyType


def test_phase11_acceptance_mcts_beats_random():
    """Acceptance: MCTS consistently beats Random baseline (>= 80% win rate)."""
    evaluator = Evaluator()
    config = MCTSConfig(
        num_simulations=25,
        max_rollout_depth=4,
        rollout_policy=RolloutPolicyType.GREEDY,
        seed=42,
    )
    mcts = MCTSAgent(config=config, name="MCTSAgent")
    random_agent = RandomAgent(name="RandomAgent")

    res = evaluator.evaluate(agent_a=mcts, agent_b=random_agent, num_games=10, seed=42)
    assert res.agent_a_win_rate >= 0.80, (
        f"MCTS win rate vs Random too low: {res.agent_a_win_rate * 100:.1f}%"
    )


def test_phase11_acceptance_mcts_beats_greedy():
    """Acceptance: MCTS consistently beats Greedy baseline (>= 60% win rate)."""
    evaluator = Evaluator()
    config = MCTSConfig(
        num_simulations=30,
        max_rollout_depth=5,
        rollout_policy=RolloutPolicyType.STRATEGIC,
        seed=42,
    )
    mcts = MCTSAgent(config=config, name="MCTSAgent")
    greedy = GreedyAgent(name="GreedyAgent")

    res = evaluator.evaluate(agent_a=mcts, agent_b=greedy, num_games=10, seed=42)
    assert res.agent_a_win_rate >= 0.60, (
        f"MCTS win rate vs Greedy too low: {res.agent_a_win_rate * 100:.1f}%"
    )


def test_phase11_acceptance_mcts_competitive_with_strategic():
    """Acceptance: MCTS is competitive with Strategic heuristic baseline (>= 50% win rate)."""
    evaluator = Evaluator()
    config = MCTSConfig(
        num_simulations=50,
        max_rollout_depth=8,
        rollout_policy=RolloutPolicyType.STRATEGIC,
        seed=42,
    )
    mcts = MCTSAgent(config=config, name="MCTSAgent")
    strategic = StrategicAgent(name="StrategicAgent")

    res = evaluator.evaluate(agent_a=mcts, agent_b=strategic, num_games=10, seed=42)
    assert res.agent_a_win_rate >= 0.50, (
        f"MCTS win rate vs Strategic too low: {res.agent_a_win_rate * 100:.1f}% (scores: MCTS={res.metrics_a.avg_score:.1f}, Strategic={res.metrics_b.avg_score:.1f})"
    )


def test_phase11_acceptance_benchmark_runner(tmp_path: Path):
    """Acceptance: MCTSBenchmarkRunner generates valid Markdown and JSON reports."""
    config = MCTSConfig(
        num_simulations=15,
        max_rollout_depth=3,
        rollout_policy=RolloutPolicyType.RANDOM,
        seed=42,
    )
    runner = MCTSBenchmarkRunner(mcts_config=config)
    results = runner.run_head_to_head_suite(num_games=2, seed=42)

    md_path = tmp_path / "phase11_report.md"
    json_path = tmp_path / "phase11_report.json"
    runner.generate_report(results, output_md_path=md_path, output_json_path=json_path)

    assert md_path.exists()
    assert json_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Fase 11: MCTS & Heuristic Rollout Benchmark Report" in content
