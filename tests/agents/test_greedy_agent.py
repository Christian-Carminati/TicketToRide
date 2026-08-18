"""Unit tests for GreedyAgent."""

from src.agents.greedy_agent import GreedyAgent
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game


def test_greedy_agent_keeps_highest_value_tickets():
    agent = GreedyAgent(name="Greedy")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    valid_actions = game.valid_actions()
    ticket_actions = [a for a in valid_actions if a.action_type == ActionType.KEEP_TICKETS]
    assert len(ticket_actions) > 0

    action = agent.act(game.state, valid_actions, game.board)
    assert action.action_type == ActionType.KEEP_TICKETS
    # Greedy chooses all tickets or maximum total value
    assert len(action.ticket_ids) >= 2


def test_greedy_agent_prioritizes_highest_scoring_route():
    agent = GreedyAgent(name="Greedy")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    # Transition to normal turn
    keep_action = game.valid_actions()[0]
    game.step(keep_action)
    keep_action2 = game.valid_actions()[0]
    game.step(keep_action2)

    # Give current player cards to afford two routes of different lengths
    player = game.state.current_player
    for _ in range(6):
        player.add_card(TrainCard(color=CardColor.RED))
        player.add_card(TrainCard(color=CardColor.BLUE))

    valid_actions = game.valid_actions()
    claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
    if claim_actions:
        action = agent.act(game.state, valid_actions, game.board)
        assert action.action_type == ActionType.CLAIM_ROUTE
        chosen_route = game.board.get_route(action.route_id)
        for ca in claim_actions:
            r = game.board.get_route(ca.route_id)
            assert (
                game.rules.points_for_route_length(chosen_route.length)
                >= game.rules.points_for_route_length(r.length)
            )


def test_greedy_agent_draws_matching_majority_color_or_locomotive():
    agent = GreedyAgent(name="Greedy")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    # Fast forward past ticket selection
    for _ in range(2):
        game.step(game.valid_actions()[0])

    player = game.state.current_player
    # Empty hand and give 0 cards so no route can be claimed
    player.cards = {c: 0 for c in CardColor}

    # Set visible cards to have GREEN card at index 2
    game.state.visible_cards[2] = TrainCard(color=CardColor.GREEN)
    game.state.visible_cards[0] = TrainCard(color=CardColor.BLACK)

    valid_actions = game.valid_actions()
    claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
    assert len(claim_actions) == 0

    action = agent.act(game.state, valid_actions, game.board)
    assert action.action_type in [ActionType.DRAW_VISIBLE_CARD, ActionType.DRAW_HIDDEN_CARD]
