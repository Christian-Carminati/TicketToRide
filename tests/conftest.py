"""Pytest shared fixtures for TicketToRide RL Lab."""

import pytest
from src.game.board import Board, City
from src.game.card import CardColor
from src.game.game import Game
from src.game.route import Route


@pytest.fixture
def sample_cities() -> dict:
    return {
        "NYC": City(id="NYC", name="New York", x=100.0, y=50.0),
        "BOS": City(id="BOS", name="Boston", x=120.0, y=40.0),
        "DC": City(id="DC", name="Washington", x=90.0, y=70.0),
    }


@pytest.fixture
def sample_board(sample_cities) -> Board:
    board = Board(cities=sample_cities)
    board.routes.append(
        Route(id="r_nyc_bos", city_a="NYC", city_b="BOS", length=2, color=CardColor.RED)
    )
    board.routes.append(
        Route(id="r_nyc_dc", city_a="NYC", city_b="DC", length=2, color=CardColor.BLUE)
    )
    return board


@pytest.fixture
def sample_game(sample_board) -> Game:
    return Game(board=sample_board, num_players=2, seed=42)
