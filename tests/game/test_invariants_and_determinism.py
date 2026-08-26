"""Property-based tests, invariants validation, and deterministic replay suite."""

import random

from src.game.game import Game
from src.game.maps import create_synthetic_mini_board


def test_deterministic_full_game_replay():
    """Verify that two games with identical seeds and policies produce bit-exact identical histories."""

    def play_game(seed: int):
        game = Game(num_players=2, seed=seed)
        game.reset(seed=seed)
        action_history = []

        rng = random.Random(seed)
        for _ in range(100):
            if game.state.is_game_over:
                break
            valid = game.valid_actions()
            if not valid:
                break
            action = rng.choice(valid)
            action_history.append(action.to_dict())
            game.step(action)

        return game.state.to_dict(), action_history

    state1, history1 = play_game(seed=999)
    state2, history2 = play_game(seed=999)

    assert history1 == history2
    assert state1 == state2


def test_card_conservation_invariant():
    """Verify that the total number of train cards is exactly conserved (110 cards) at every step."""
    game = Game(num_players=2, seed=777)
    game.reset(seed=777)
    rng = random.Random(777)

    for _ in range(100):
        if game.state.is_game_over:
            break

        total_in_hands = sum(p.total_cards() for p in game.state.players)
        total_visible = len(game.state.visible_cards)
        total_deck = len(game.state.train_deck)
        total_discard = len(game.state.discard_pile)

        assert total_in_hands + total_visible + total_deck + total_discard == 110, (
            f"Cards broken: hands={total_in_hands}, vis={total_visible}, deck={total_deck}, disc={total_discard}"
        )

        valid = game.valid_actions()
        if not valid:
            break
        game.step(rng.choice(valid))


def test_train_conservation_invariant():
    """Verify that trains remaining + spent trains == 45 per player at every step."""
    game = Game(num_players=2, seed=555)
    game.reset(seed=555)
    rng = random.Random(555)

    for _ in range(100):
        if game.state.is_game_over:
            break

        for p in game.state.players:
            claimed_routes = [game.board.get_route(r_id) for r_id in p.claimed_route_ids]
            spent_trains = sum(r.length for r in claimed_routes if r is not None)
            assert p.trains_remaining + spent_trains == 45, (
                f"Train invariant broken for player {p.id}: {p.trains_remaining} + {spent_trains} != 45"
            )

        valid = game.valid_actions()
        if not valid:
            break
        game.step(rng.choice(valid))


def test_full_game_simulation_to_completion():
    """Simulate a game to full completion by setting reduced train counts."""
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2, seed=123)
    game.reset(seed=123)

    # Set each player to 4 trains so claiming 1 route of length 2 leaves 2 trains (triggering last round)
    for p in game.state.players:
        p.trains_remaining = 4

    rng = random.Random(123)

    for _ in range(150):
        if game.state.is_game_over:
            break
        valid = game.valid_actions()
        if not valid:
            break
        # Bias choice toward claiming routes if available
        claim_actions = [a for a in valid if a.action_type.value == "claim_route"]
        if claim_actions:
            action = rng.choice(claim_actions)
        else:
            action = rng.choice(valid)
        game.step(action)

    assert game.state.is_game_over is True
    assert game.state.winner_id in ["player_0", "player_1"]
    for p in game.state.players:
        assert isinstance(p.score, int)
