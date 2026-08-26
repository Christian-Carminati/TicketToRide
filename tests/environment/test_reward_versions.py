"""Unit tests for Phase 7 modular reward versions, component breakdowns, and factory."""

import pytest
from src.environment.reward import (
    CustomRewardCalculator,
    RewardFactory,
    RewardV1_Sparse,
    RewardV2_DenseRoutes,
    RewardV3_TicketMilestones,
    RewardV4_StrategicShaped,
    RewardWeights,
)
from src.game.action import Action, ActionType
from src.game.game import Game
from src.game.maps import create_synthetic_mini_board


def test_reward_factory_aliases():
    board, _ = create_synthetic_mini_board()
    r1 = RewardFactory.create("sparse", board=board)
    assert isinstance(r1, RewardV1_Sparse)

    r1_int = RewardFactory.create(1, board=board)
    assert isinstance(r1_int, RewardV1_Sparse)

    r2 = RewardFactory.create("dense_routes", board=board)
    assert isinstance(r2, RewardV2_DenseRoutes)

    r3 = RewardFactory.create("ticket_milestones", board=board)
    assert isinstance(r3, RewardV3_TicketMilestones)

    r4 = RewardFactory.create("strategic", board=board)
    assert isinstance(r4, RewardV4_StrategicShaped)

    rc = RewardFactory.create("custom", board=board, weights=RewardWeights(win_bonus=50.0))
    assert isinstance(rc, CustomRewardCalculator)


def test_reward_v1_sparse_intermediate_and_terminal():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)
    calc = RewardV1_Sparse(board=board)

    prev = game.state
    # Draw card action (intermediate step)
    action = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
    game.step(action)
    next_s = game.state

    step_reward = calc.calculate(prev, action, next_s, player_index=0)
    components = calc.get_components(prev, action, next_s, player_index=0)
    assert step_reward == 0.0
    assert components["step_reward"] == 0.0

    # Force game over to verify terminal reward
    next_s.is_game_over = True
    next_s.winner_id = next_s.players[0].id
    next_s.players[0].score = 25
    next_s.players[1].score = 10

    term_reward = calc.calculate(prev, action, next_s, player_index=0)
    term_comp = calc.get_components(prev, action, next_s, player_index=0)
    assert term_reward > 0.0
    assert term_comp["win_bonus"] > 0.0
    assert pytest.approx(sum(term_comp.values())) == term_reward


def test_reward_v2_dense_routes_claim_reward():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)
    calc = RewardV2_DenseRoutes(board=board, weights=RewardWeights(step_penalty=0.01))

    # Give player 0 enough cards and claim route
    p = game.state.players[0]
    p.cards["red"] = 5
    route = board.routes[0]
    action = Action(action_type=ActionType.CLAIM_ROUTE, route_id=route.id, color_chosen="red")
    prev = game.state
    game.step(action)
    next_s = game.state

    reward = calc.calculate(prev, action, next_s, player_index=0)
    components = calc.get_components(prev, action, next_s, player_index=0)
    assert reward > 0.0
    assert components["route_points"] > 0.0
    assert components["step_penalty"] < 0.0
    assert pytest.approx(sum(components.values())) == reward


def test_reward_v3_ticket_milestones():
    board, tickets = create_synthetic_mini_board()
    calc = RewardV3_TicketMilestones(board=board)
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)

    # Claim a route
    p = game.state.players[0]
    p.cards["red"] = 10
    route = board.routes[0]
    action = Action(action_type=ActionType.CLAIM_ROUTE, route_id=route.id, color_chosen="red")
    prev = game.state
    game.step(action)
    next_s = game.state

    reward = calc.calculate(prev, action, next_s, player_index=0)
    components = calc.get_components(prev, action, next_s, player_index=0)
    assert "ticket_completion" in components
    assert "route_points" in components
    assert pytest.approx(sum(components.values())) == reward


def test_reward_v4_strategic_shaped():
    board, tickets = create_synthetic_mini_board()
    calc = RewardV4_StrategicShaped(board=board)
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)

    prev = game.state
    action = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
    game.step(action)
    next_s = game.state

    reward = calc.calculate(prev, action, next_s, player_index=0)
    components = calc.get_components(prev, action, next_s, player_index=0)
    assert pytest.approx(sum(components.values())) == reward


def test_custom_reward_calculator_weights():
    board, _tickets = create_synthetic_mini_board()
    weights = RewardWeights(
        route_points_weight=3.5,
        step_penalty=0.05,
        win_bonus=100.0,
    )
    calc = CustomRewardCalculator(weights=weights, board=board)
    assert calc.weights.route_points_weight == 3.5
    assert calc.weights.win_bonus == 100.0
