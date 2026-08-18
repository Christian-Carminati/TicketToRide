"""Unit tests for Player state, cards inventory, and route affordability."""

from src.game.card import CardColor, TrainCard
from src.game.player import Player
from src.game.route import Route
from src.game.ticket import DestinationTicket


def test_player_card_management():
    player = Player(id="p0", name="Player 1")
    assert player.total_cards() == 0
    player.add_card(TrainCard(color=CardColor.RED))
    player.add_card(TrainCard(color=CardColor.LOCOMOTIVE))
    assert player.total_cards() == 2
    assert player.cards[CardColor.RED] == 1
    assert player.cards[CardColor.LOCOMOTIVE] == 1

    # Remove cards
    success = player.remove_cards({CardColor.RED: 1})
    assert success is True
    assert player.cards[CardColor.RED] == 0

    # Cannot remove more cards than possessed
    fail = player.remove_cards({CardColor.RED: 1})
    assert fail is False


def test_player_afford_route():
    player = Player(id="p0", name="Player 1", trains_remaining=4)
    route_blue = Route(id="r1", city_a="A", city_b="B", length=3, color=CardColor.BLUE)

    # Cannot afford with 0 cards
    assert player.can_afford_route(route_blue) == []

    # Add 2 Blue and 1 Locomotive
    player.add_card(TrainCard(color=CardColor.BLUE))
    player.add_card(TrainCard(color=CardColor.BLUE))
    player.add_card(TrainCard(color=CardColor.LOCOMOTIVE))

    options = player.can_afford_route(route_blue)
    assert len(options) == 1
    assert options[0] == {CardColor.BLUE: 2, CardColor.LOCOMOTIVE: 1}

    # Test gray route with multiple possible payment color combinations
    route_gray = Route(id="r2", city_a="A", city_b="B", length=2, color=None)
    player.add_card(TrainCard(color=CardColor.RED))
    # Player has 2 Blue, 1 Locomotive, 1 Red.
    # Can pay 2 length with: {Blue: 2}, {Blue: 1, Locomotive: 1}, {Red: 1, Locomotive: 1}, {Locomotive: 2} (not enough locos)
    gray_options = player.can_afford_route(route_gray)
    assert len(gray_options) >= 3


def test_player_has_trains():
    player = Player(id="p0", name="Player 1", trains_remaining=2)
    route_short = Route(id="r1", city_a="A", city_b="B", length=2)
    route_long = Route(id="r2", city_a="A", city_b="B", length=3)
    assert player.has_trains_for_route(route_short) is True
    assert player.has_trains_for_route(route_long) is False


def test_player_serialization():
    player = Player(id="p0", name="Player 1", score=15, trains_remaining=30)
    player.tickets.append(DestinationTicket(id="t1", city_a="A", city_b="B", points=5))
    player.pending_tickets.append(DestinationTicket(id="t2", city_a="C", city_b="D", points=10))
    player.add_card(TrainCard(color=CardColor.GREEN))
    player.claimed_route_ids.append("r1")

    data = player.to_dict()
    restored = Player.from_dict(data)

    assert restored.id == "p0"
    assert restored.name == "Player 1"
    assert restored.score == 15
    assert restored.trains_remaining == 30
    assert restored.cards[CardColor.GREEN] == 1
    assert len(restored.tickets) == 1
    assert restored.tickets[0].id == "t1"
    assert len(restored.pending_tickets) == 1
    assert restored.pending_tickets[0].id == "t2"
    assert restored.claimed_route_ids == ["r1"]
