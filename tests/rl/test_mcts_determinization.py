"""Unit tests for POMDP determinization in MCTS."""

from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.random import SeededRNG
from src.rl.mcts_determinization import determinize_game


def test_determinization_preserves_root_player():
    """Verify that root player's private inventory is untouched by determinization."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    root_p0 = game.state.players[0]
    original_cards_p0 = dict(root_p0.cards)
    original_tickets_p0 = list(root_p0.tickets)

    rng = SeededRNG(12345)
    det_game = determinize_game(game, root_player_id="player_0", rng=rng)

    det_p0 = det_game.state.players[0]
    # Root player's hand and tickets must be strictly preserved
    assert det_p0.cards == original_cards_p0
    assert det_p0.tickets == original_tickets_p0


def test_determinization_preserves_card_counts_and_public_visible():
    """Verify that opponent total card count and public visible cards are unchanged."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    opp = game.state.players[1]
    opp_total_cards = opp.total_cards()
    deck_len = len(game.state.train_deck)
    visible_colors = [c.color for c in game.state.visible_cards]

    rng = SeededRNG(999)
    det_game = determinize_game(game, root_player_id="player_0", rng=rng)

    det_opp = det_game.state.players[1]
    assert det_opp.total_cards() == opp_total_cards
    assert len(det_game.state.train_deck) == deck_len
    assert [c.color for c in det_game.state.visible_cards] == visible_colors


def test_determinization_conserves_total_hidden_cards():
    """Verify that all hidden cards across opponents and deck are strictly conserved."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    opp = game.state.players[1]
    original_hidden_pool = sum(opp.cards.values()) + len(game.state.train_deck)

    rng = SeededRNG(555)
    det_game = determinize_game(game, root_player_id="player_0", rng=rng)

    det_hidden_pool = sum(det_game.state.players[1].cards.values()) + len(det_game.state.train_deck)
    assert det_hidden_pool == original_hidden_pool
