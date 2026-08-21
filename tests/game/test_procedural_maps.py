"""Tests for Procedural Map Generator and Map Split infrastructure."""

from collections import deque
import pytest
from src.game.card import CardColor
from src.game.procedural import ProceduralMapConfig, ProceduralMapGenerator


def test_procedural_map_determinism():
    config = ProceduralMapConfig(num_cities=8, num_routes=14, num_tickets=10)
    gen = ProceduralMapGenerator(config)

    board1, tickets1 = gen.generate(seed=42)
    board2, tickets2 = gen.generate(seed=42)

    assert len(board1.cities) == 8
    assert len(board1.routes) == 14
    assert len(tickets1) == 10

    assert [c.id for c in board1.cities.values()] == [c.id for c in board2.cities.values()]
    assert [(r.city_a, r.city_b, r.length, r.color) for r in board1.routes] == [
        (r.city_a, r.city_b, r.length, r.color) for r in board2.routes
    ]
    assert [(t.city_a, t.city_b, t.points) for t in tickets1] == [
        (t.city_a, t.city_b, t.points) for t in tickets2
    ]


def test_procedural_map_graph_connectivity():
    gen = ProceduralMapGenerator()
    for seed in [1, 7, 42, 100, 999]:
        board, tickets = gen.generate(seed=seed)
        cities_list = list(board.cities.values())
        assert len(cities_list) > 0

        # Build adjacency list
        adj: dict[str, list[str]] = {c.name: [] for c in cities_list}
        for r in board.routes:
            adj[r.city_a].append(r.city_b)
            adj[r.city_b].append(r.city_a)

        # BFS from city 0
        start_city = cities_list[0].name
        visited = set()
        queue = deque([start_city])
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    queue.append(neighbor)

        # 100% of cities must be reachable (connected graph)
        assert len(visited) == len(cities_list), f"Graph disconnected on seed {seed}"


def test_procedural_map_valid_attributes():
    gen = ProceduralMapGenerator()
    board, tickets = gen.generate(seed=123)

    for r in board.routes:
        assert 1 <= r.length <= 6
        assert r.color is None or isinstance(r.color, CardColor)
        assert board.get_city(r.city_a) is not None
        assert board.get_city(r.city_b) is not None

    city_names = set(board.cities.keys())
    for t in tickets:
        assert t.city_a in city_names
        assert t.city_b in city_names
        assert t.city_a != t.city_b
        assert t.points >= 2


def test_procedural_map_dataset_split():
    from src.game.procedural import MapSplit, ProceduralMapDataset

    gen = ProceduralMapGenerator()
    dataset = ProceduralMapDataset(gen)

    split = dataset.create_split(
        train_seeds=[1, 2, 3],
        val_seeds=[10, 11],
        test_seeds=[100, 101, 102, 103],
    )

    assert isinstance(split, MapSplit)
    assert len(split.train_maps) == 3
    assert len(split.val_maps) == 2
    assert len(split.test_maps) == 4

    # Verify boards are unique and populated
    train_city_counts = [len(b.cities) for b, t in split.train_maps]
    assert all(count == 8 for count in train_city_counts)

