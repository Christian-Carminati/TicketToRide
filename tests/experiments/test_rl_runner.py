import os
import tempfile

from src.agents.random_agent import RandomAgent
from src.experiments.config import ExperimentConfig
from src.experiments.evaluator import MultiOpponentEvaluator
from src.experiments.runner import ExperimentRunner
from src.game.maps import create_synthetic_mini_board


def test_multi_opponent_evaluator_basic() -> None:
    board, _ = create_synthetic_mini_board()
    evaluator = MultiOpponentEvaluator(board=board, seed=42)
    agent = RandomAgent(name="EvalCandidate")

    results = evaluator.evaluate(agent=agent, opponents=["random", "greedy"], games_per_opponent=4)

    assert "win_rate_vs_random" in results
    assert "win_rate_vs_greedy" in results
    assert "score_diff_vs_random" in results


def test_experiment_runner_dqn_training() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_dqn_run",
            seed=42,
            environment={"board": "mini", "players": 2},
            algorithm={"name": "dqn", "learning_starts": 10, "batch_size": 16, "buffer_size": 500},
            training={
                "total_timesteps": 50,
                "batch_size": 16,
                "eval_freq": 25,
                "eval_episodes_per_opponent": 2,
                "checkpoint_dir": tmpdir,
            },
            evaluation={"opponents": ["random"]},
        )

        runner = ExperimentRunner(config)
        record = runner.run()

        assert record.name == "test_dqn_run"
        assert record.algorithm == "dqn"
        assert os.path.exists(os.path.join(tmpdir, "test_dqn_run_latest.pt"))


def test_experiment_runner_ppo_training() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_ppo_run",
            seed=42,
            environment={"board": "mini", "players": 2},
            algorithm={"name": "ppo"},
            training={
                "total_timesteps": 64,
                "rollout_steps": 32,
                "num_epochs": 1,
                "batch_size": 16,
                "eval_freq": 32,
                "eval_episodes_per_opponent": 2,
                "checkpoint_dir": tmpdir,
            },
            evaluation={"opponents": ["random"]},
        )

        runner = ExperimentRunner(config)
        record = runner.run()

        assert record.name == "test_ppo_run"
        assert record.algorithm == "ppo"
        assert os.path.exists(os.path.join(tmpdir, "test_ppo_run_latest.pt"))
