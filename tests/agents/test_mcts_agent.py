"""Unit tests for MCTSAgent compliance with BaseAgent and Gymnasium."""

import numpy as np
from src.agents.mcts_agent import MCTSAgent
from src.game.game import Game
from src.rl.mcts import MCTSConfig, RolloutPolicyType


def test_mcts_agent_act_compliance():
    """Verify MCTSAgent.act returns a valid action from Game state."""
    agent = MCTSAgent(
        config=MCTSConfig(
            num_simulations=10,
            max_rollout_depth=3,
            rollout_policy=RolloutPolicyType.RANDOM,
        )
    )
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    valid_actions = game.valid_actions()
    action = agent.act(game.state, valid_actions, game.board)

    assert action in valid_actions


def test_mcts_agent_deterministic_repeatability():
    """Verify identical seeds produce identical action sequences."""
    config = MCTSConfig(num_simulations=20, max_rollout_depth=4, seed=12345)
    agent1 = MCTSAgent(config=config, name="MCTS1")
    agent2 = MCTSAgent(config=config, name="MCTS2")

    game1 = Game(num_players=2, seed=42)
    game1.reset(seed=42)
    game2 = Game(num_players=2, seed=42)
    game2.reset(seed=42)

    a1 = agent1.act(game1.state, game1.valid_actions(), game1.board)
    a2 = agent2.act(game2.state, game2.valid_actions(), game2.board)

    assert a1 == a2


def test_mcts_agent_select_action_mask():
    """Verify Gymnasium select_action obeys action masking."""
    agent = MCTSAgent(config=MCTSConfig(num_simulations=5))
    obs = np.zeros(10)
    mask = np.array([0, 0, 1, 0, 1])
    idx = agent.select_action(obs, action_mask=mask)
    assert idx in [2, 4]
