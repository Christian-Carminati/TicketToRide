"""Unit tests for Board, Routes, and Map loaders."""

from src.game.board import Board, City
from src.game.card import CardColor
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.game.route import Route


def test_synthetic_mini_board():
    board, tickets = create_synthetic_mini_board()
    assert len(board.cities) >= 5
    assert len(board.routes) >= 6
    assert len(tickets) >= 3


def test_usa_board_completeness():
    board, tickets = load_usa_board()
    assert len(board.cities) == 36
    assert len(board.routes) == 100
    assert len(tickets) == 30

    # Test double route query between Boston and New York
    routes_bos_nyc = board.get_routes_between("Boston", "New York")
    assert len(routes_bos_nyc) == 2
    assert routes_bos_nyc[0].double_route_pair_id == routes_bos_nyc[1].id
    assert routes_bos_nyc[1].double_route_pair_id == routes_bos_nyc[0].id
    assert routes_bos_nyc[0].is_double_route is True


def test_board_adjacency_and_queries():
    board, _ = create_synthetic_mini_board()
    cities = list(board.cities.keys())
    adj = board.get_adjacent_cities(cities[0])
    assert len(adj) > 0

    route = board.routes[0]
    fetched = board.get_route(route.id)
    assert fetched is not None
    assert fetched.id == route.id
