"""Unit tests for Baseline and RL Agent interfaces and RandomAgent."""

import numpy as np
from src.agents.random_agent import RandomAgent
from src.game.game import Game


def test_random_agent_act_deterministic():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    valid_actions = game.valid_actions()
    assert len(valid_actions) > 0

    agent1 = RandomAgent(seed=123, name="Random_1")
    agent2 = RandomAgent(seed=123, name="Random_2")

    a1 = agent1.act(game.state, valid_actions, game.board)
    a2 = agent2.act(game.state, valid_actions, game.board)
    assert a1 == a2
    assert a1 in valid_actions


def test_random_agent_select_action_gym():
    agent = RandomAgent(seed=42)
    obs = np.zeros(10)
    mask = np.array([False, True, False, True, False])
    chosen = agent.select_action(obs, action_mask=mask)
    assert chosen in [1, 3]


def test_random_agent_reset():
    agent = RandomAgent(seed=42)
    agent.reset(seed=999)
    assert agent.rng is not None


def test_mixed_opponent_agent():
    from src.agents.mixed_agent import MixedOpponentAgent

    mixed = MixedOpponentAgent(seed=42)
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    valid_actions = game.valid_actions()

    act = mixed.act(game.state, valid_actions, game.board)
    assert act in valid_actions

    # Test reset switches / initializes properly
    mixed.reset(seed=123)
    assert mixed.active_agent is not None
    assert "MixedBot" in mixed.name

