"""Unit tests for RecurrentPPOAgent with persistent LSTM memory."""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.agents.recurrent_ppo_agent import RecurrentPPOAgent
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import ObservationV1
from src.game.action import Action, ActionType
from src.game.game import Game
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.rl.lstm_ppo import RecurrentMaskedActorCritic


def test_recurrent_ppo_agent_act_and_state_evolution():
    board, tickets = load_usa_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)

    obs_encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)
    obs_dim = obs_encoder.observation_shape[0]
    action_space = DiscreteActionSpace(board=board)
    action_dim = action_space.n

    model = RecurrentMaskedActorCritic(input_dim=obs_dim, action_dim=action_dim, hidden_dim=64, lstm_hidden_dim=64)
    agent = RecurrentPPOAgent(model=model, board=board, tickets=tickets, num_players=2)

    valid_actions = game.valid_actions()
    action = agent.act(game.state, valid_actions, board)
    assert action in valid_actions

    # Check that hidden state updated
    h, c = agent.current_hidden
    assert not torch.all(h == 0.0)

    # Test reset clears hidden state
    agent.reset()
    h_reset, c_reset = agent.current_hidden
    assert torch.all(h_reset == 0.0)
    assert torch.all(c_reset == 0.0)


def test_recurrent_ppo_agent_empty_valid_actions_raises():
    agent = RecurrentPPOAgent()
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    with pytest.raises(ValueError, match="No valid actions"):
        agent.act(game.state, [], game.board)


def test_recurrent_ppo_agent_checkpoint_loading_and_saving():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "recurrent_ppo.pt")
        board, tickets = create_synthetic_mini_board()
        action_space = DiscreteActionSpace(board=board)
        obs_encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)
        obs_dim = obs_encoder.observation_shape[0]
        action_dim = action_space.n

        model = RecurrentMaskedActorCritic(
            input_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            lstm_hidden_dim=32,
        )
        agent1 = RecurrentPPOAgent(
            model=model,
            board=board,
            tickets=tickets,
            num_players=2,
            deterministic=True,
        )
        agent1.save(ckpt_path)

        agent2 = RecurrentPPOAgent(
            model=ckpt_path,
            board=board,
            tickets=tickets,
            num_players=2,
            deterministic=True,
        )

        obs = np.random.randn(obs_dim).astype(np.float32)
        mask = np.ones(action_dim, dtype=bool)

        a1 = agent1.select_action(obs, mask)
        a2 = agent2.select_action(obs, mask)
        assert a1 == a2
        assert isinstance(a1, int)


def test_recurrent_ppo_agent_select_action_stochastic():
    board, tickets = create_synthetic_mini_board()
    agent = RecurrentPPOAgent(board=board, tickets=tickets, num_players=2, deterministic=False)
    obs_dim = agent.obs_encoder.observation_shape[0]
    action_dim = agent.action_space.n

    obs = np.zeros(obs_dim, dtype=np.float32)
    mask = np.zeros(action_dim, dtype=bool)
    mask[1] = True
    mask[2] = True

    action = agent.select_action(obs, mask, deterministic=False)
    assert action in (1, 2)


def test_recurrent_ppo_agent_fallback_on_invalid_decoded_action():
    board, tickets = create_synthetic_mini_board()
    agent = RecurrentPPOAgent(board=board, tickets=tickets, num_players=2)
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)

    # Valid actions in initial state
    valid_actions = [Action(action_type=ActionType.DRAW_HIDDEN_CARD)]
    # Mock select_action to return an index that doesn't decode to DRAW_HIDDEN_CARD
    agent.select_action = lambda obs, mask, deterministic=None: 999999 if 999999 < len(mask) else len(mask) - 1

    act = agent.act(game.state, valid_actions, board)
    assert act == valid_actions[0]
