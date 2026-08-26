"""Tests for DiscreteActionSpace bijective mapping."""

from src.environment.action_space import DiscreteActionSpace
from src.game.action import Action, ActionType
from src.game.card import CardColor
from src.game.maps import create_synthetic_mini_board, load_usa_board


def test_discrete_action_space_mini_board_bijection():
    board, _ = create_synthetic_mini_board()
    space = DiscreteActionSpace(board=board)

    # 1 (hidden) + 5 (visible) + 1 (draw tickets) + 7 (keep tickets) + routes
    assert space.n > 0
    for action_id in range(space.n):
        action = space.to_action(action_id)
        assert isinstance(action, Action)
        recovered_id = space.to_id(action)
        assert recovered_id == action_id, f"Failed roundtrip for action {action}"


def test_discrete_action_space_usa_board_coverage():
    board, _ = load_usa_board()
    space = DiscreteActionSpace(board=board)

    # Draw hidden
    draw_hidden = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
    assert space.to_id(draw_hidden) == 0

    # Draw visible
    for slot in range(5):
        draw_vis = Action(action_type=ActionType.DRAW_VISIBLE_CARD, card_index=slot)
        assert space.to_id(draw_vis) == 1 + slot

    # Draw tickets
    draw_tix = Action(action_type=ActionType.DRAW_TICKETS)
    assert space.to_id(draw_tix) == 6

    # All keep tickets combinations (7 non-empty subsets)
    for subset_idx in range(7):
        act_id = 7 + subset_idx
        act = space.to_action(act_id)
        assert act.action_type == ActionType.KEEP_TICKETS

    # Claim routes (colored + gray)
    route_claim_count = 0
    for r in board.routes:
        if r.color == CardColor.LOCOMOTIVE or r.color is None:
            route_claim_count += 8  # Gray route has 8 colored actions
        else:
            route_claim_count += 1
    assert space.n == 14 + route_claim_count
