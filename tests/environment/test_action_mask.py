"""Tests for ActionMasker legal action mask generation."""

import numpy as np
import pytest

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.game.action import Action, ActionType
from src.game.card import CardColor
from src.game.game import Game
from src.game.maps import create_synthetic_mini_board
from src.game.state import TurnState


def test_action_masker_initial_tickets_state():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)
    space = DiscreteActionSpace(board=board)
    masker = ActionMasker(space)

    valid_actions = game.valid_actions()
    pending = state.current_player.pending_tickets
    mask = masker.compute_mask(valid_actions, pending_tickets=pending)

    assert mask.shape == (space.n,)
    assert mask.dtype == bool
    assert np.any(mask)

    # In CHOOSING_INITIAL_TICKETS, only keep_tickets with >= 2 tickets should be True
    # Actions 0..6 (draws) must be False
    assert not np.any(mask[0:7])
    # Action 14+ (claim routes) must be False
    assert not np.any(mask[14:])


def test_action_masker_normal_turn():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)
    space = DiscreteActionSpace(board=board)
    masker = ActionMasker(space)

    # Transition to NORMAL turn state
    act0 = game.valid_actions()[0]
    game.step(act0)
    act1 = game.valid_actions()[0]
    game.step(act1)
    assert game.state.turn_state == TurnState.NORMAL

    valid_actions = game.valid_actions()
    mask = masker.compute_mask(valid_actions, pending_tickets=game.state.current_player.pending_tickets)

    assert mask.shape == (space.n,)
    assert np.any(mask)
    # Hidden card draw (index 0) must be True
    assert mask[0] is True or mask[0] == True
    # Keep tickets actions (indices 7..13) must be False
    assert not np.any(mask[7:14])
