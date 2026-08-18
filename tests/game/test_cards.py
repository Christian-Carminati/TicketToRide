"""Unit tests for Train Cards, Decks, and Seeded RNG."""

from src.game.card import CardColor, create_standard_train_deck
from src.game.random import SeededRNG
from src.game.ticket import DestinationTicket


def test_standard_train_deck_composition():
    deck = create_standard_train_deck()
    assert len(deck) == 110
    locomotives = [c for c in deck if c.is_locomotive()]
    assert len(locomotives) == 14
    for color in CardColor:
        if color != CardColor.LOCOMOTIVE:
            color_cards = [c for c in deck if c.color == color]
            assert len(color_cards) == 12


def test_seeded_rng_reproducibility():
    rng1 = SeededRNG(seed=12345)
    rng2 = SeededRNG(seed=12345)
    deck1 = create_standard_train_deck()
    deck2 = create_standard_train_deck()
    rng1.shuffle(deck1)
    rng2.shuffle(deck2)
    assert [c.color.value for c in deck1] == [c.color.value for c in deck2]

    # Sample reproducibility
    s1 = rng1.sample(deck1, 5)
    s2 = rng2.sample(deck2, 5)
    assert [c.color.value for c in s1] == [c.color.value for c in s2]


def test_destination_ticket_creation():
    ticket = DestinationTicket(id="t_nyc_mia", city_a="New York", city_b="Miami", points=10)
    assert ticket.points == 10
    assert ticket.city_a == "New York"
    assert ticket.city_b == "Miami"
