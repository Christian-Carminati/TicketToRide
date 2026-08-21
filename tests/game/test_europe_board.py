"""Tests for official Ticket to Ride Europe board."""

import pytest
from src.game.game import Game
from src.game.maps import load_europe_board


def test_load_europe_board_structure():
    board, tickets = load_europe_board()
    assert len(board.cities) >= 40
    assert len(board.routes) >= 90
    assert len(tickets) >= 40

    # Ensure valid city references
    city_names = set(board.cities.keys())
    for r in board.routes:
        assert r.city_a in city_names, f"Unknown city_a: {r.city_a}"
        assert r.city_b in city_names, f"Unknown city_b: {r.city_b}"
        assert 1 <= r.length <= 8

    for t in tickets:
        assert t.city_a in city_names, f"Unknown ticket city_a: {t.city_a}"
        assert t.city_b in city_names, f"Unknown ticket city_b: {t.city_b}"
        assert t.points >= 4


def test_europe_board_gameplay():
    board, tickets = load_europe_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2, seed=42)
    state = game.reset(seed=42)

    assert not state.is_game_over
    assert len(state.players) == 2
    assert len(state.visible_cards) == 5
    valid_actions = game.valid_actions()
    assert len(valid_actions) > 0
