"""Tests for ObservationV1 encoder and POMDP anti-leakage invariants."""

import numpy as np
import pytest

from src.environment.observation import ObservationV1
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.game.ticket import DestinationTicket


def test_observation_v1_mini_board_shape_and_range():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)

    encoder = ObservationV1(board=board, initial_tickets=tickets)
    obs = encoder.encode(state, player_index=0)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == encoder.observation_shape
    # D_mini = 9 (hand) + 50 (visible) + 2 (player) + 18 (6 routes*3) + 21 (7 tickets*3) + 4 (opp) + 5 (decks) + 4 (phase) = 113
    assert obs.shape == (113,)
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_observation_v1_usa_board_shape():
    board, tickets = load_usa_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)

    encoder = ObservationV1(board=board, initial_tickets=tickets)
    obs = encoder.encode(state, player_index=0)

    # D_usa = 9 + 50 + 2 + (100*3=300) + (30*3=90) + 4 + 5 + 4 = 464
    assert obs.shape == (464,)
    assert obs.shape == encoder.observation_shape
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_observation_v1_anti_leakage_guarantee():
    """Modifying opponent hidden card colors or deck order (preserving count) must NOT change player 0 observation."""
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=123)
    state = game.reset(seed=123)

    encoder = ObservationV1(board=board, initial_tickets=tickets)
    obs_original = encoder.encode(state, player_index=0).copy()

    # Modify opponent hidden hand card colors while preserving exact total card count (4 cards)
    opp_total_before = sum(state.players[1].cards.values())
    for color in CardColor:
        state.players[1].cards[color] = 0
    state.players[1].cards[CardColor.RED] = opp_total_before
    assert sum(state.players[1].cards.values()) == opp_total_before

    # Modify hidden train deck sequence without changing length
    if len(state.train_deck) >= 2:
        state.train_deck[0], state.train_deck[1] = state.train_deck[1], state.train_deck[0]

    obs_modified = encoder.encode(state, player_index=0)
    np.testing.assert_array_equal(obs_original, obs_modified)
