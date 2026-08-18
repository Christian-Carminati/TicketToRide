"""Unit tests for GameRules, action legality, and double-route constraints."""

from src.game.action import ActionType
from src.game.card import CardColor, TrainCard
from src.game.maps import create_synthetic_mini_board
from src.game.player import Player
from src.game.rules import GameRules
from src.game.state import GameState, TurnState
from src.game.ticket import DestinationTicket


def test_3_locomotives_flush_rule():
    visible_3_locos = [
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.RED),
        TrainCard(color=CardColor.BLUE),
    ]
    assert GameRules.should_flush_visible_cards(visible_3_locos) is True

    visible_2_locos = [
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.GREEN),
        TrainCard(color=CardColor.RED),
        TrainCard(color=CardColor.BLUE),
    ]
    assert GameRules.should_flush_visible_cards(visible_2_locos) is False


def test_double_route_blocking_2_players():
    board, _ = create_synthetic_mini_board()
    double_r = [r for r in board.routes if r.is_double_route]
    assert len(double_r) >= 2
    r1 = double_r[0]
    r2 = board.get_route(r1.double_route_pair_id)
    r1.claimed_by = "p0"

    player1 = Player(id="p1", name="P2", trains_remaining=10)
    state = GameState(num_players=2)

    # In 2-player game, r2 should not be claimable because parallel r1 is claimed
    assert (
        GameRules.can_claim_route(r2, player1, state, board, num_players=2) is False
    )

    # In 4-player game, r2 IS claimable by a different player (player1)
    player1.add_card(TrainCard(color=CardColor.BLUE))
    player1.add_card(TrainCard(color=CardColor.BLUE))
    assert GameRules.can_claim_route(r2, player1, state, board, num_players=4) is True

    # But player0 who claimed r1 cannot claim r2 in any player count
    player0 = Player(id="p0", name="P1", trains_remaining=10)
    player0.add_card(TrainCard(color=CardColor.BLUE))
    player0.add_card(TrainCard(color=CardColor.BLUE))
    assert GameRules.can_claim_route(r2, player0, state, board, num_players=4) is False


def test_valid_actions_in_drawing_second_card_state():
    state = GameState(
        turn_state=TurnState.DRAWING_SECOND_CARD,
        visible_cards=[
            TrainCard(color=CardColor.RED),
            TrainCard(color=CardColor.LOCOMOTIVE),
        ],
        train_deck=[TrainCard(color=CardColor.BLUE)],
    )
    player = Player(id="p0", name="P1")
    board, _ = create_synthetic_mini_board()
    actions = GameRules.get_valid_actions(player, state, board, num_players=2)

    # In DRAWING_SECOND_CARD, drawing visible locomotive is illegal
    assert any(a.action_type == ActionType.DRAW_HIDDEN_CARD for a in actions)
    assert any(
        a.action_type == ActionType.DRAW_VISIBLE_CARD and a.card_index == 0
        for a in actions
    )
    assert not any(
        a.action_type == ActionType.DRAW_VISIBLE_CARD and a.card_index == 1
        for a in actions
    )
    assert not any(a.action_type == ActionType.CLAIM_ROUTE for a in actions)


def test_valid_actions_in_choosing_tickets_state():
    t1 = DestinationTicket(id="t1", city_a="A", city_b="B", points=5)
    t2 = DestinationTicket(id="t2", city_a="C", city_b="D", points=7)
    t3 = DestinationTicket(id="t3", city_a="E", city_b="F", points=9)

    player = Player(id="p0", name="P1", pending_tickets=[t1, t2, t3])
    board, _ = create_synthetic_mini_board()

    # Initial tickets: must keep at least 2
    state_initial = GameState(turn_state=TurnState.CHOOSING_INITIAL_TICKETS)
    actions_init = GameRules.get_valid_actions(
        player, state_initial, board, num_players=2
    )
    assert len(actions_init) == 4  # 3 combinations of 2 + 1 combination of 3
    for a in actions_init:
        assert a.action_type == ActionType.KEEP_TICKETS
        assert len(a.ticket_ids) >= 2

    # Mid-game tickets: must keep at least 1
    state_mid = GameState(turn_state=TurnState.CHOOSING_TICKETS)
    actions_mid = GameRules.get_valid_actions(player, state_mid, board, num_players=2)
    assert len(actions_mid) == 7  # 3 of 1 + 3 of 2 + 1 of 3
    for a in actions_mid:
        assert a.action_type == ActionType.KEEP_TICKETS
        assert len(a.ticket_ids) >= 1
