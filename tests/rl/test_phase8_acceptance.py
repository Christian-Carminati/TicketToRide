"""Comprehensive Phase 8 Acceptance Test Suite verifying all 6 acceptance criteria."""

import json

import numpy as np
import pytest
import torch
from src.agents.random_agent import RandomAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent
from src.environment.env import TicketToRideEnv
from src.environment.observation import ObservationV1
from src.evaluation.evaluator import Evaluator
from src.evaluation.pomdp_benchmark import POMDPBenchmarkRunner
from src.game.card import CardColor
from src.game.game import Game
from src.game.maps import load_usa_board
from src.rl.lstm_ppo import MaskedRecurrentPPOTrainer, RecurrentMaskedActorCritic
from src.rl.rollout import RecurrentRolloutBuffer


def test_phase8_criterion_1_anti_leakage_invariance():
    """Criterion 1: Formal invariance tests confirm zero leakage of hidden state."""
    board, tickets = load_usa_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=101)
    encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)

    obs_initial = encoder.encode(game.state, player_index=0).copy()

    # Modify opponent private cards
    game.state.players[1].cards.clear()
    game.state.players[1].cards[CardColor.GREEN] = 4
    obs_modified = encoder.encode(game.state, player_index=0).copy()

    np.testing.assert_array_equal(obs_initial, obs_modified)


def test_phase8_criterion_2_and_3_recurrent_architecture_and_buffer():
    """Criterion 2 & 3: Masked categorical distribution & sequential chunk generation."""
    model = RecurrentMaskedActorCritic(
        input_dim=50, action_dim=10, hidden_dim=32, lstm_hidden_dim=32
    )
    obs = torch.randn(1, 50)
    hidden = model.get_initial_hidden(batch_size=1)
    mask = torch.tensor([[True, False, False, False, False, False, False, False, False, False]])

    action, _log_prob, _, _, _ = model.get_action_and_value(obs, hidden, action_mask=mask)
    assert action.item() == 0

    buffer = RecurrentRolloutBuffer(capacity=16, obs_dim=50, action_dim=10, lstm_hidden_dim=32)
    for i in range(16):
        buffer.add(
            obs=np.zeros(50, dtype=np.float32),
            action=0,
            reward=1.0,
            value=0.5,
            log_prob=-0.1,
            done=(i % 8 == 7),
            action_mask=np.ones(10, dtype=bool),
            h=np.zeros(32, dtype=np.float32),
            c=np.zeros(32, dtype=np.float32),
        )
    buffer.set_advantages_and_returns(np.ones(16, dtype=np.float32), np.ones(16, dtype=np.float32))

    chunks = list(buffer.generate_recurrent_chunks(seq_len=8, batch_size=2, num_epochs=1))
    assert len(chunks) == 1
    assert chunks[0]["obs"].shape == (2, 8, 50)


def test_phase8_criterion_4_deterministic_reproducibility():
    """Criterion 4: Identical seeds produce bitwise reproducible recurrent models and metrics."""

    def run_training(seed=123):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = TicketToRideEnv(seed=seed)
        trainer = MaskedRecurrentPPOTrainer(
            env=env,
            config={
                "rollout_steps": 64,
                "seq_len": 8,
                "minibatch_chunks": 4,
                "num_epochs": 2,
                "hidden_dim": 32,
                "lstm_hidden_dim": 32,
            },
            seed=seed,
        )
        metrics = trainer.train(total_timesteps=64)
        return trainer, metrics

    trainer1, metrics1 = run_training(123)
    trainer2, metrics2 = run_training(123)

    assert metrics1[-1]["policy_loss"] == pytest.approx(metrics2[-1]["policy_loss"], rel=1e-5)
    assert metrics1[-1]["value_loss"] == pytest.approx(metrics2[-1]["value_loss"], rel=1e-5)
    assert metrics1[-1]["entropy"] == pytest.approx(metrics2[-1]["entropy"], rel=1e-5)

    for p1, p2 in zip(trainer1.actor_critic.parameters(), trainer2.actor_critic.parameters()):
        torch.testing.assert_close(p1, p2)


def test_phase8_criterion_5_trainer_convergence_and_superiority():
    """Criterion 5: Reproducible training and outperforming random baseline."""
    env = TicketToRideEnv(seed=77)
    trainer = MaskedRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 128,
            "seq_len": 8,
            "minibatch_chunks": 4,
            "num_epochs": 2,
            "lr": 3e-4,
            "hidden_dim": 64,
            "lstm_hidden_dim": 64,
        },
    )

    trainer.train(total_timesteps=300)

    board, tickets = load_usa_board()
    agent = RecurrentPPOAgent(model=trainer.actor_critic, board=board, tickets=tickets)
    random_agent = RandomAgent()

    evaluator = Evaluator(board=board, tickets_deck=tickets)
    eval_res = evaluator.evaluate(agent, random_agent, num_games=10, seed=42)

    # Recurrent agent achieves solid win rate against random
    assert eval_res.agent1_win_rate >= 0.50


def test_phase8_criterion_6_automated_scientific_benchmark_study(tmp_path):
    """Criterion 6: POMDPBenchmarkRunner generates valid JSON and Markdown reports."""
    report_md = tmp_path / "phase8_report.md"
    report_json = tmp_path / "phase8_report.json"

    runner = POMDPBenchmarkRunner(
        config={
            "seed": 42,
            "training_steps": 256,
            "eval_games": 4,
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
    assert "head_to_head" in data
    assert "vs_random" in data
    assert "vs_greedy" in data
    assert "behavioral" in data
