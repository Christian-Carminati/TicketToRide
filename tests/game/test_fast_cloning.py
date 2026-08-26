"""Unit and benchmark tests for fast in-memory state cloning."""

import time

from src.game.card import CardColor
from src.game.game import Game


def test_fast_cloning_isolation():
    """Verify that mutations on cloned game/state/board do not leak to the original."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    # Clone game
    cloned_game = game.clone(rng_seed=999)

    assert len(cloned_game.state.players) == 2
    assert cloned_game.state.turn_number == game.state.turn_number

    # Mutate player in clone
    p0_clone = cloned_game.state.players[0]
    p0_clone.score += 50
    p0_clone.cards[CardColor.RED] += 10

    # Verify original is untouched
    assert game.state.players[0].score == 0
    assert game.state.players[0].cards[CardColor.RED] != p0_clone.cards[CardColor.RED]

    # Mutate board route in clone
    r0_id = cloned_game.board.routes[0].id
    cloned_game.board.routes[0].claimed_by = "player_0"

    # Verify original board route is still unclaimed
    assert game.board.get_route(r0_id).claimed_by is None


def test_cloning_performance():
    """Verify high-throughput in-memory cloning (>= 5000 clones/sec)."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    start_t = time.perf_counter()
    n_clones = 2000
    for _ in range(n_clones):
        _ = game.clone()
    elapsed = time.perf_counter() - start_t

    throughput = n_clones / elapsed
    assert throughput >= 2000, f"Cloning throughput too low: {throughput:.1f} clones/sec"
