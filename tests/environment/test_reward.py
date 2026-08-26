"""Tests for DefaultRewardCalculator (RewardV1)."""

import pytest
from src.environment.reward import DefaultRewardCalculator, RewardWeights
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.maps import create_synthetic_mini_board


def test_reward_route_points_and_step_penalty():
    board, tickets = create_synthetic_mini_board()
    weights = RewardWeights(route_points_weight=1.0, step_penalty=0.05)
    calc = DefaultRewardCalculator(weights=weights, board=board)

    game = Game(board=board, tickets_deck=tickets, seed=42)
    game.reset(seed=42)

    # Initial tickets setup
    game.step(game.valid_actions()[0])
    game.step(game.valid_actions()[0])

    prev_state = game.state
    # Give player 0 cards to claim r_ab_1 (length 2 -> 2 points)
    player0 = game.state.players[0]
    player0.add_card(TrainCard(color=CardColor.RED))
    player0.add_card(TrainCard(color=CardColor.RED))

    claim_act = Action(
        action_type=ActionType.CLAIM_ROUTE,
        route_id="r_ab_1",
        color_chosen=CardColor.RED,
    )
    next_state = game.step(claim_act)

    reward = calc.calculate(
        prev_state=prev_state,
        action=claim_act,
        next_state=next_state,
        player_index=0,
    )

    # Delta score is +2, step penalty is -0.05 -> reward = 1.95
    assert pytest.approx(reward, 1e-4) == 1.95


def test_reward_terminal_win_bonus():
    board, tickets = create_synthetic_mini_board()
    weights = RewardWeights(win_bonus=20.0, loss_penalty=10.0, score_diff_weight=0.5)
    calc = DefaultRewardCalculator(weights=weights, board=board)

    game = Game(board=board, tickets_deck=tickets, seed=42)
    prev_state = game.reset(seed=42)

    # Mock game over state with player 0 winner
    next_state = Game(board=board, tickets_deck=tickets, seed=42).reset(seed=42)
    next_state.is_game_over = True
    next_state.winner_id = "player_0"
    next_state.players[0].score = 30
    next_state.players[1].score = 10

    reward = calc.calculate(
        prev_state=prev_state,
        action=Action(action_type=ActionType.DRAW_HIDDEN_CARD),
        next_state=next_state,
        player_index=0,
    )

    # Win bonus (20) + score diff (0.5 * (30 - 10) = 10) = 30.0
    assert reward >= 30.0
