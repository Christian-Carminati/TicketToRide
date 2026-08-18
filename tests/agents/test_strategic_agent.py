"""Unit tests for StrategicHeuristicAgent."""

import pytest

from src.agents.strategic_agent import StrategicHeuristicAgent
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.ticket import DestinationTicket


def test_strategic_agent_initial_ticket_synergy():
    agent = StrategicHeuristicAgent(name="Strategic")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    valid_actions = game.valid_actions()
    action = agent.act(game.state, valid_actions, game.board)
    assert action.action_type == ActionType.KEEP_TICKETS
    # Agent keeps at least 2 synergistic tickets
    assert len(action.ticket_ids) >= 2


def test_strategic_agent_targets_shortest_path_route():
    agent = StrategicHeuristicAgent(name="Strategic")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    # Transition past ticket selection
    for _ in range(2):
        act = game.valid_actions()[0]
        game.step(act)

    player = game.state.current_player
    # Give player plenty of all cards
    for c in CardColor:
        for _ in range(10):
            player.add_card(TrainCard(color=c))

    valid_actions = game.valid_actions()
    claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
    assert len(claim_actions) > 0

    action = agent.act(game.state, valid_actions, game.board)
    assert action.action_type == ActionType.CLAIM_ROUTE
    assert action.route_id is not None


def test_strategic_agent_dijkstra_routing_blocked_routes():
    agent = StrategicHeuristicAgent(name="Strategic")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    # Fast forward ticket selection
    for _ in range(2):
        game.step(game.valid_actions()[0])

    player = game.state.players[0]
    opponent = game.state.players[1]

    # Create a test ticket from Vancouver to Seattle
    ticket = DestinationTicket(id="test_van_sea", city_a="Vancouver", city_b="Seattle", points=5)
    player.tickets = [ticket]

    # Opponent claims both routes between Vancouver and Seattle
    routes_van_sea = game.board.get_routes_between("Vancouver", "Seattle")
    for r in routes_van_sea:
        r.claimed_by = opponent.id

    # The shortest path should now detour around Seattle or report no path if completely blocked
    path = agent._compute_shortest_path_routes(player, ticket, game.board)
    # Check that none of the routes in path are claimed by opponent
    for r in path:
        assert r.claimed_by != opponent.id
