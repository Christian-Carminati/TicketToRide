# tests/environment/test_pomdp_anti_leakage.py
import numpy as np
import pytest

from src.environment.observation import ObservationV1
from src.game.card import Card, CardColor
from src.game.game import Game
from src.game.maps import load_usa_board
from src.game.ticket import DestinationTicket


def _create_standard_game() -> tuple[Game, ObservationV1]:
    board, tickets = load_usa_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)
    obs_encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)
    return game, obs_encoder


def test_pomdp_invariant_opponent_cards_color_distribution():
    """Modifying opponent card colors while keeping count constant must produce identical observations."""
    game, obs_encoder = _create_standard_game()

    # Base observation for Player 0
    obs_original = obs_encoder.encode(game.state, player_index=0).copy()

    # Modify Player 1 (opponent) cards: keep total = 4, but switch all to RED
    game.state.players[1].cards.clear()
    game.state.players[1].cards[CardColor.RED] = 4

    obs_modified_red = obs_encoder.encode(game.state, player_index=0).copy()
    np.testing.assert_array_equal(obs_original, obs_modified_red)

    # Switch all opponent cards to LOCOMOTIVE
    game.state.players[1].cards.clear()
    game.state.players[1].cards[CardColor.LOCOMOTIVE] = 4

    obs_modified_loco = obs_encoder.encode(game.state, player_index=0).copy()
    np.testing.assert_array_equal(obs_original, obs_modified_loco)


def test_pomdp_invariant_opponent_destination_tickets():
    """Altering opponent destination tickets must produce zero changes in Player 0 observation."""
    game, obs_encoder = _create_standard_game()
    obs_original = obs_encoder.encode(game.state, player_index=0).copy()

    # Add extra arbitrary tickets to opponent
    fake_ticket = DestinationTicket(id="fake_t", city_a="Boston", city_b="Miami", points=22)
    game.state.players[1].tickets.append(fake_ticket)

    obs_modified_tickets = obs_encoder.encode(game.state, player_index=0).copy()
    np.testing.assert_array_equal(obs_original, obs_modified_tickets)

    # Clear opponent tickets completely
    game.state.players[1].tickets.clear()
    obs_empty_tickets = obs_encoder.encode(game.state, player_index=0).copy()
    np.testing.assert_array_equal(obs_original, obs_empty_tickets)


def test_pomdp_invariant_hidden_deck_order():
    """Permuting/shuffling face-down train deck must produce zero changes in Player 0 observation."""
    game, obs_encoder = _create_standard_game()
    obs_original = obs_encoder.encode(game.state, player_index=0).copy()

    # Reverse train deck order
    game.state.train_deck.reverse()
    obs_reversed_deck = obs_encoder.encode(game.state, player_index=0).copy()
    np.testing.assert_array_equal(obs_original, obs_reversed_deck)

    # Replace deck with arbitrary colors preserving length
    deck_len = len(game.state.train_deck)
    game.state.train_deck = [Card(color=CardColor.BLUE) for _ in range(deck_len)]
    obs_blue_deck = obs_encoder.encode(game.state, player_index=0).copy()
    np.testing.assert_array_equal(obs_original, obs_blue_deck)


def test_pomdp_sensitivity_to_public_visible_cards():
    """Modifying face-up visible cards on the table MUST produce a strictly localized change in observation."""
    game, obs_encoder = _create_standard_game()
    obs_original = obs_encoder.encode(game.state, player_index=0).copy()

    # Change slot 0 visible card to a guaranteed different color
    curr_color = game.state.visible_cards[0].color
    new_color = CardColor.RED if curr_color != CardColor.RED else CardColor.BLUE
    game.state.visible_cards[0] = Card(color=new_color)
    obs_modified = obs_encoder.encode(game.state, player_index=0).copy()

    # Must NOT be identical
    assert not np.array_equal(obs_original, obs_modified)

    # Only visible cards section (indices 9 to 59) should differ
    diff_indices = np.where(obs_original != obs_modified)[0]
    assert len(diff_indices) > 0
    assert all(9 <= idx < 59 for idx in diff_indices)
