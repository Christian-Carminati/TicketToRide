"""Phase 6 Acceptance Test Suite: PPO From Scratch, CleanRL Details & Scientific Benchmark."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.greedy_agent import GreedyAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.benchmark import PPOBenchmarkRunner
from src.evaluation.evaluator import Evaluator
from src.game.maps import create_synthetic_mini_board
from src.rl.ppo import MaskedPPOTrainer


def test_phase6_ppo_cleanrl_training_and_acceptance() -> None:
    """Verify that custom CleanRL PPO trains effectively and outperforms RandomAgent."""
    torch.manual_seed(42)
    np.random.seed(42)

    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(
        board=board,
        tickets_deck=tickets,
        opponent=RandomAgent(seed=42),
    )
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "lr": 1e-3,
            "anneal_lr": False,
            "clip_vloss": False,
            "target_kl": 0.04,
            "rollout_steps": 256,
            "num_epochs": 4,
            "minibatch_size": 32,
            "orthogonal_init": True,
        },
    )

    # Train for 6,400 environment timesteps
    train_summary = trainer.train(total_timesteps=6400)
    assert train_summary["total_timesteps"] >= 6400

    agent = PPOAgent(
        name="PPO_Phase6",
        actor_critic=trainer.actor_critic,
        encoder=env.encoder,
        discrete_actions=env.discrete_actions,
        device=trainer.device,
    )
    random_agent = RandomAgent(seed=456, name="RandomOpponent")

    evaluator = Evaluator(board=board, tickets_deck=tickets, seed=123)
    results = evaluator.evaluate(agent_a=agent, agent_b=random_agent, num_games=40, seed=123)

    m_ppo = results["PPO_Phase6"]
    m_rand = results["RandomOpponent"]

    # PPO must reliably beat random agent with high win rate and positive score differential
    assert m_ppo.win_rate >= 0.65
    assert m_ppo.avg_score > m_rand.avg_score


def test_phase6_deterministic_reproducibility() -> None:
    """Verify that two PPO trainers with identical seeds produce identical losses and model weights."""
    board, tickets = create_synthetic_mini_board()

    def run_training_run(seed: int) -> tuple[dict[str, Any], list[np.ndarray]]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=RandomAgent(seed=seed),
        )
        trainer = MaskedPPOTrainer(
            env=env,
            config={
                "rollout_steps": 64,
                "num_epochs": 2,
                "minibatch_size": 16,
                "orthogonal_init": True,
                "anneal_lr": True,
                "clip_vloss": True,
            },
        )
        trainer.collect_rollout()
        metrics = trainer.train_epoch()
        weights = [p.clone().detach().cpu().numpy() for p in trainer.actor_critic.parameters()]
        return metrics, weights

    m1, w1 = run_training_run(seed=999)
    m2, w2 = run_training_run(seed=999)

    assert np.isclose(m1["policy_loss"], m2["policy_loss"], atol=1e-5)
    assert np.isclose(m1["value_loss"], m2["value_loss"], atol=1e-5)

    # Check network weights match bitwise
    for p1, p2 in zip(w1, w2, strict=True):
        assert np.allclose(p1, p2, atol=1e-6)



def test_phase6_cleanrl_ablation_and_report_generation(tmp_path: Path) -> None:
    """Verify that the automated benchmark runner generates a valid ablation study and markdown report."""
    json_file = tmp_path / "phase6_report.json"
    md_file = tmp_path / "phase6_report.md"

    runner = PPOBenchmarkRunner(
        board_type="mini",
        games_per_opponent=5,
        total_training_steps=400,
        output_json=str(json_file),
        output_md=str(md_file),
        seed=101,
    )

    results = runner.run_benchmark(run_ablation=True, ablation_steps=200, ablation_games=3)

    assert results["board_type"] == "mini"
    assert "opponents" in results
    assert "ablation" in results
    assert len(results["ablation"]) == 4

    assert json_file.exists()
    assert md_file.exists()

    md_text = md_file.read_text(encoding="utf-8")
    assert "# PPO Benchmark & Ablation Report" in md_text
    assert "ppo_full" in md_text
    assert "ppo_no_ortho" in md_text
    assert "ppo_no_vf_clip" in md_text
    assert "ppo_no_lr_anneal" in md_text
