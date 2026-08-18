"""Unit tests for Graph algorithms and Game engine orchestration."""

from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.graph import check_ticket_completed, compute_longest_continuous_path
from src.game.maps import create_synthetic_mini_board
from src.game.player import Player
from src.game.route import Route
from src.game.ticket import DestinationTicket


def test_graph_ticket_connectivity():
    routes = [
        Route(id="r1", city_a="City_A", city_b="City_B", length=2, claimed_by="p0"),
        Route(id="r2", city_a="City_B", city_b="City_C", length=3, claimed_by="p0"),
    ]
    t_connected = DestinationTicket(
        id="t1", city_a="City_A", city_b="City_C", points=5
    )
    t_not_connected = DestinationTicket(
        id="t2", city_a="City_A", city_b="City_D", points=7
    )

    assert check_ticket_completed(routes, t_connected) is True
    assert check_ticket_completed(routes, t_not_connected) is False


def test_longest_continuous_path():
    routes = [
        Route(id="r1", city_a="City_A", city_b="City_B", length=2),
        Route(id="r2", city_a="City_B", city_b="City_C", length=4),
        Route(id="r3", city_a="City_C", city_b="City_D", length=1),
    ]
    assert compute_longest_continuous_path(routes) == 7


def test_full_game_reset_and_initial_tickets():
    game = Game(num_players=2, seed=42)
    state = game.reset()
    assert len(state.players) == 2
    assert len(state.visible_cards) == 5
    assert state.players[0].total_cards() == 4
    assert state.players[1].total_cards() == 4
    assert len(state.players[0].pending_tickets) == 3


def test_game_step_draw_cards():
    game = Game(num_players=2, seed=42)
    game.reset()

    # Player 0 keeps 2 initial tickets
    p0_pending = game.state.players[0].pending_tickets
    act1 = Action(
        action_type=ActionType.KEEP_TICKETS,
        ticket_ids=(p0_pending[0].id, p0_pending[1].id),
    )
    game.step(act1)
    assert len(game.state.players[0].tickets) == 2

    # Player 1 keeps 2 initial tickets
    p1_pending = game.state.players[1].pending_tickets
    act2 = Action(
        action_type=ActionType.KEEP_TICKETS,
        ticket_ids=(p1_pending[0].id, p1_pending[1].id),
    )
    game.step(act2)
    assert len(game.state.players[1].tickets) == 2

    # Now turn is NORMAL for Player 0
    p0_cards_before = game.state.players[0].total_cards()
    game.step(Action(action_type=ActionType.DRAW_HIDDEN_CARD))
    game.step(Action(action_type=ActionType.DRAW_HIDDEN_CARD))
    assert game.state.players[0].total_cards() == p0_cards_before + 2
    # Turn advanced to Player 1
    assert game.state.current_player_index == 1


def test_game_step_claim_route():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2, seed=42)
    game.reset()

    # Skip initial tickets
    p0 = game.state.players[0]
    p1 = game.state.players[1]
    game.step(
        Action(
            action_type=ActionType.KEEP_TICKETS,
            ticket_ids=(p0.pending_tickets[0].id, p0.pending_tickets[1].id),
        )
    )
    game.step(
        Action(
            action_type=ActionType.KEEP_TICKETS,
            ticket_ids=(p1.pending_tickets[0].id, p1.pending_tickets[1].id),
        )
    )

    # Give Player 0 enough cards to claim r_ab_1 (length 2, RED)
    p0.add_card(TrainCard(color=CardColor.RED))
    p0.add_card(TrainCard(color=CardColor.RED))
    initial_score = p0.score
    initial_trains = p0.trains_remaining

    claim_act = Action(
        action_type=ActionType.CLAIM_ROUTE,
        route_id="r_ab_1",
        color_chosen=CardColor.RED,
        locomotives_count=0,
    )
    game.step(claim_act)

    assert p0.score == initial_score + 2  # 2 points for length 2
    assert p0.trains_remaining == initial_trains - 2
    assert "r_ab_1" in p0.claimed_route_ids
    assert board.get_route("r_ab_1").claimed_by == "player_0"
    # Turn advanced to Player 1
    assert game.state.current_player_index == 1
