"""Phase 4 Acceptance Test Suite: DQN, PPO, and Multi-Opponent Benchmarks."""

import numpy as np
import torch
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.evaluator import Evaluator
from src.experiments.evaluator import MultiOpponentEvaluator
from src.game.maps import create_synthetic_mini_board
from src.rl.dqn import MaskedDQNTrainer
from src.rl.ppo import MaskedPPOTrainer


def test_phase4_dqn_beats_random_acceptance() -> None:
    """Acceptance Criterion 1: Masked Double-DQN learns to beat RandomAgent with >= 75% win rate."""
    torch.manual_seed(100)
    np.random.seed(100)

    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(
        board=board,
        tickets_deck=tickets,
        opponent=RandomAgent(seed=100),
    )

    config = {
        "lr": 5e-4,
        "gamma": 0.99,
        "buffer_size": 20000,
        "batch_size": 64,
        "target_update_freq": 200,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 6000,
        "learning_starts": 200,
    }

    trainer = MaskedDQNTrainer(env=env, config=config)

    # Train on Mini Map (10,000 steps)
    for _ in range(10000):
        trainer.step()
        trainer.train_step()

    agent = DQNAgent(
        input_dim=env.observation_space.shape[0],
        action_dim=int(env.action_space.n),
        encoder=env.encoder,
        discrete_actions=env.discrete_actions,
    )
    agent.q_net.load_state_dict(trainer.policy_net.state_dict())

    evaluator = Evaluator(board=board, tickets_deck=tickets, seed=123)
    results = evaluator.evaluate_head_to_head(
        agent_a=agent,
        agent_b=RandomAgent(seed=456, name="RandomOpponent"),
        num_games=50,
    )

    print(
        f"\nDQN vs Random win rate: {results['agent_a_win_rate']:.2f}, score: {results['agent_a_mean_score']:.1f} vs {results['agent_b_mean_score']:.1f}"
    )
    assert results["agent_a_win_rate"] >= 0.65, (
        f"DQN win rate {results['agent_a_win_rate']} must be >= 0.65"
    )


def test_phase4_ppo_beats_random_acceptance() -> None:
    """Acceptance Criterion 2: Masked PPO learns to beat RandomAgent with >= 75% win rate."""
    torch.manual_seed(42)
    np.random.seed(42)

    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(
        board=board,
        tickets_deck=tickets,
        opponent=RandomAgent(seed=42),
    )

    config = {
        "lr": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "rollout_steps": 256,
        "num_epochs": 4,
        "minibatch_size": 32,
    }

    trainer = MaskedPPOTrainer(env=env, config=config)

    # Train for 25 rollouts (~6,400 steps on Mini Map)
    for _ in range(25):
        trainer.collect_rollout()
        trainer.train_epoch()

    agent = PPOAgent(
        input_dim=env.observation_space.shape[0],
        action_dim=int(env.action_space.n),
        encoder=env.encoder,
        discrete_actions=env.discrete_actions,
    )
    agent.actor_critic.load_state_dict(trainer.actor_critic.state_dict())

    evaluator = Evaluator(board=board, tickets_deck=tickets, seed=123)
    results = evaluator.evaluate_head_to_head(
        agent_a=agent,
        agent_b=RandomAgent(seed=456, name="RandomOpponent"),
        num_games=40,
    )

    print(
        f"\nPPO vs Random win rate: {results['agent_a_win_rate']:.2f}, score: {results['agent_a_mean_score']:.1f} vs {results['agent_b_mean_score']:.1f}"
    )
    assert results["agent_a_win_rate"] >= 0.65, (
        f"PPO win rate {results['agent_a_win_rate']} must be >= 0.65"
    )


def test_phase4_multi_opponent_evaluation_suite() -> None:
    """Acceptance Criterion 3: Evaluator outputs full benchmark metrics against all baselines."""
    board, tickets = create_synthetic_mini_board()
    evaluator = MultiOpponentEvaluator(board=board, tickets_deck=tickets, seed=42)
    agent = RandomAgent(name="TestAgent")

    metrics = evaluator.evaluate(
        agent=agent,
        opponents=["random", "greedy", "strategic"],
        games_per_opponent=10,
    )
    for opp in ["random", "greedy", "strategic"]:
        assert f"win_rate_vs_{opp}" in metrics
        assert f"score_diff_vs_{opp}" in metrics
        assert f"mean_score_vs_{opp}" in metrics


def test_phase4_deterministic_reproducibility() -> None:
    """Acceptance Criterion 4: Deterministic training reproducibility given same random seed."""
    board, tickets = create_synthetic_mini_board()

    def run_training_run(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=RandomAgent(seed=seed),
        )
        trainer = MaskedPPOTrainer(
            env=env,
            config={"rollout_steps": 64, "num_epochs": 2, "minibatch_size": 16, "lr": 1e-3},
        )
        trainer.collect_rollout()
        metrics = trainer.train_epoch()
        weights = [p.clone().detach().numpy() for p in trainer.actor_critic.parameters()]
        return metrics, weights

    m1, w1 = run_training_run(seed=999)
    m2, w2 = run_training_run(seed=999)

    assert np.isclose(m1["policy_loss"], m2["policy_loss"], atol=1e-6)
    assert np.isclose(m1["value_loss"], m2["value_loss"], atol=1e-6)
    for p1, p2 in zip(w1, w2):
        assert np.allclose(p1, p2, atol=1e-6)
