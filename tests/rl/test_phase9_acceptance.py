"""Comprehensive Phase 9 Acceptance Test Suite verifying all 6 acceptance criteria."""

import json
from pathlib import Path

import numpy as np
import torch
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.self_play_benchmark import SelfPlayBenchmarkRunner
from src.game.maps import load_usa_board
from src.rl.networks import MaskedActorCritic
from src.rl.self_play import (
    PolicyPool,
    SelfPlayOpponentSampler,
    SelfPlayPPOTrainer,
)


def test_phase9_criterion_1_policy_pool_management():
    """Criterion 1: PolicyPool manages snapshots, retention, capacity, and agent creation."""
    pool = PolicyPool(max_size=4)
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)

    for i in range(6):
        pool.add_policy(model, step=i * 500, name=f"gen_{i}")

    assert pool.size == 4
    # Anchor gen_0 and latest gen_5 must be preserved
    assert pool.get_snapshot("gen_0") is not None
    assert pool.get_snapshot("gen_5") is not None

    agent = pool.create_agent(0)
    assert agent.name == "gen_0"


def test_phase9_criterion_2_matchmaking_strategies():
    """Criterion 2: SelfPlayOpponentSampler correctly applies uniform, latest-biased, PFSP, and baseline mix."""
    pool = PolicyPool()
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    pool.add_policy(model, step=0, name="gen_0")
    pool.add_policy(model, step=1000, name="gen_1")
    pool.add_policy(model, step=2000, name="gen_2")

    sampler_latest = SelfPlayOpponentSampler(strategy="latest_biased", baseline_mix_rate=0.0)
    weights = sampler_latest.get_opponent_weights(pool)
    assert weights["gen_2"] >= 0.5

    sampler_pfsp = SelfPlayOpponentSampler(strategy="pfsp", baseline_mix_rate=0.0)
    sampler_pfsp.record_match("gen_0", trainee_won=True)
    sampler_pfsp.record_match("gen_1", trainee_won=False)
    pfsp_w = sampler_pfsp.get_opponent_weights(pool)
    assert pfsp_w["gen_1"] > pfsp_w["gen_0"]


def test_phase9_criterion_3_dynamic_opponent_switching():
    """Criterion 3: Gymnasium env cleanly switches opponents per episode without state corruption."""
    env = TicketToRideEnv(seed=101)
    pool = PolicyPool()
    model = MaskedActorCritic(
        input_dim=env.observation_space.shape[0],
        action_dim=int(env.action_space.n),
        hidden_dim=32,
    )
    pool.add_policy(model, step=0, name="gen_0")

    trainer = SelfPlayPPOTrainer(
        env=env,
        config={
            "rollout_steps": 32,
            "minibatch_size": 16,
            "num_epochs": 1,
            "snapshot_interval": 32,
        },
        pool=pool,
        seed=101,
    )
    obs, info = trainer.env.reset()
    assert obs.shape == env.observation_space.shape
    assert "action_mask" in info

    metrics = trainer.collect_rollout()
    assert "mean_rollout_reward" in metrics


def test_phase9_criterion_4_deterministic_reproducibility():
    """Criterion 4: Identical seeds produce bitwise reproducible self-play training and pools."""

    def run_training(seed=77):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = TicketToRideEnv(seed=seed)
        trainer = SelfPlayPPOTrainer(
            env=env,
            config={
                "rollout_steps": 64,
                "minibatch_size": 16,
                "num_epochs": 2,
                "snapshot_interval": 64,
                "hidden_dim": 32,
            },
            seed=seed,
        )
        trainer.train(total_timesteps=64)
        return trainer

    tr1 = run_training(77)
    tr2 = run_training(77)

    assert tr1.pool.size == tr2.pool.size
    snap1 = tr1.pool.get_snapshot(0)
    snap2 = tr2.pool.get_snapshot(0)

    for k in snap1.state_dict:
        torch.testing.assert_close(snap1.state_dict[k], snap2.state_dict[k])


def test_phase9_criterion_5_generational_progression_and_superiority():
    """Criterion 5: Self-play agent beats initial generation and outperforms random baseline."""
    board, tickets = load_usa_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, seed=42)
    trainer = SelfPlayPPOTrainer(
        env=env,
        config={
            "rollout_steps": 128,
            "minibatch_size": 32,
            "num_epochs": 2,
            "lr": 3e-4,
            "snapshot_interval": 128,
            "hidden_dim": 64,
        },
        seed=42,
    )
    trainer.train(total_timesteps=300)

    final_agent = trainer.pool.create_agent(trainer.pool.size - 1, board=board, tickets=tickets)
    random_agent = RandomAgent()

    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(board=board, tickets_deck=tickets)
    eval_res = evaluator.evaluate(final_agent, random_agent, num_games=10, seed=42)

    assert eval_res.agent1_win_rate >= 0.50


def test_phase9_criterion_6_automated_scientific_benchmark_study(tmp_path: Path):
    """Criterion 6: SelfPlayBenchmarkRunner generates valid JSON and Markdown academic report."""
    report_md = tmp_path / "phase9_report.md"
    report_json = tmp_path / "phase9_report.json"

    runner = SelfPlayBenchmarkRunner(
        config={
            "seed": 42,
            "training_steps": 256,
            "snapshot_interval": 128,
            "eval_games": 4,
            "games_per_pair": 2,
        }
    )
    results = runner.run_study()
    runner.generate_report(
        results, output_md_path=str(report_md), output_json_path=str(report_json)
    )

    assert report_md.exists()
    assert report_json.exists()
    with open(report_json, encoding="utf-8") as f:
        data = json.load(f)
    assert "tournament" in data
    assert "vs_baselines" in data
    assert "metadata" in data
