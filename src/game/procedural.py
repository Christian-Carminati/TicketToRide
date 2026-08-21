"""Procedural map generator for Ticket to Ride environments and generalization studies."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from src.game.board import Board, City
from src.game.card import CardColor
from src.game.route import Route
from src.game.ticket import DestinationTicket

STANDARD_COLORS = [
    CardColor.PURPLE,
    CardColor.WHITE,
    CardColor.BLUE,
    CardColor.YELLOW,
    CardColor.ORANGE,
    CardColor.BLACK,
    CardColor.RED,
    CardColor.GREEN,
    None,  # Gray
]


@dataclass
class ProceduralMapConfig:
    num_cities: int = 8
    num_routes: int = 14
    num_tickets: int = 10
    allow_double_routes: bool = False
    min_city_distance: float = 0.15


class ProceduralMapGenerator:
    """Deterministic procedural map generator with connectivity guarantees."""

    def __init__(self, config: ProceduralMapConfig | None = None) -> None:
        self.config = config or ProceduralMapConfig()

    def generate(self, seed: int) -> tuple[Board, list[DestinationTicket]]:
        rng = random.Random(seed)
        board = Board()

        # 1. Generate cities with minimum pairwise distance
        cities = self._generate_cities(rng)
        for c in cities:
            board.add_city(c)

        # 2. Build connected topology using MST + additional nearest neighbor edges
        routes = self._generate_routes(cities, rng)
        board.routes = routes
        board.rebuild_indexes()

        # 3. Generate tickets based on shortest paths
        tickets = self._generate_tickets(board, cities, rng)

        return board, tickets

    def _generate_cities(self, rng: random.Random) -> list[City]:
        cities: list[City] = []
        max_attempts = 1000

        for i in range(self.config.num_cities):
            placed = False
            for _ in range(max_attempts):
                x = round(rng.uniform(0.05, 0.95), 4)
                y = round(rng.uniform(0.05, 0.95), 4)

                # Check minimum distance with existing cities
                too_close = False
                for existing in cities:
                    dist = math.hypot(x - existing.x, y - existing.y)
                    if dist < self.config.min_city_distance:
                        too_close = True
                        break

                if not too_close or len(cities) == 0:
                    city_id = f"c_{i}"
                    city_name = f"City_{chr(65 + i) if i < 26 else str(i)}"
                    cities.append(City(id=city_id, name=city_name, x=x, y=y))
                    placed = True
                    break

            if not placed:
                # Fallback if space is tight
                city_id = f"c_{i}"
                city_name = f"City_{chr(65 + i) if i < 26 else str(i)}"
                x = round(0.1 + (0.8 / max(1, self.config.num_cities)) * i, 4)
                y = round(rng.uniform(0.1, 0.9), 4)
                cities.append(City(id=city_id, name=city_name, x=x, y=y))

        return cities

    def _generate_routes(
        self, cities: list[City], rng: random.Random
    ) -> list[Route]:
        n = len(cities)
        all_edges: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.hypot(cities[i].x - cities[j].x, cities[i].y - cities[j].y)
                all_edges.append((dist, i, j))

        all_edges.sort(key=lambda x: x[0])

        # Kruskal's MST to ensure 100% connectivity
        parent = list(range(n))

        def find(u: int) -> int:
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u: int, v: int) -> bool:
            root_u, root_v = find(u), find(v)
            if root_u == root_v:
                return False
            parent[root_u] = root_v
            return True

        selected_pairs: set[tuple[int, int]] = set()
        for dist, u, v in all_edges:
            if union(u, v):
                pair = (min(u, v), max(u, v))
                selected_pairs.add(pair)

        # Add additional nearest edges up to num_routes
        for dist, u, v in all_edges:
            if len(selected_pairs) >= self.config.num_routes:
                break
            pair = (min(u, v), max(u, v))
            selected_pairs.add(pair)

        # Construct Route objects
        routes: list[Route] = []
        sorted_pairs = sorted(list(selected_pairs))

        for idx, (u, v) in enumerate(sorted_pairs):
            c_a = cities[u]
            c_b = cities[v]
            dist = math.hypot(c_a.x - c_b.x, c_a.y - c_b.y)
            # Map distance [0.15, 1.0] to length [1..6]
            length = max(1, min(6, int(round(dist * 6.5))))
            color = STANDARD_COLORS[idx % len(STANDARD_COLORS)]

            r_id = f"r_{idx}_{c_a.id}_{c_b.id}"
            route = Route(
                id=r_id,
                city_a=c_a.name,
                city_b=c_b.name,
                length=length,
                color=color,
            )
            routes.append(route)

        # Handle double routes if requested
        if self.config.allow_double_routes and len(routes) > 1:
            r1 = routes[0]
            r2_color = STANDARD_COLORS[(len(routes)) % len(STANDARD_COLORS)]
            r2 = Route(
                id=f"{r1.id}_d",
                city_a=r1.city_a,
                city_b=r1.city_b,
                length=r1.length,
                color=r2_color,
                double_route_pair_id=r1.id,
            )
            r1.double_route_pair_id = r2.id
            routes.append(r2)

        return routes

    def _generate_tickets(
        self, board: Board, cities: list[City], rng: random.Random
    ) -> list[DestinationTicket]:
        # Compute all-pairs shortest path in train length
        adj: dict[str, list[tuple[str, int]]] = {c.name: [] for c in cities}
        for r in board.routes:
            adj[r.city_a].append((r.city_b, r.length))
            adj[r.city_b].append((r.city_a, r.length))

        city_names = [c.name for c in cities]
        valid_ticket_pairs: list[tuple[str, str, int]] = []

        for i in range(len(city_names)):
            for j in range(i + 1, len(city_names)):
                src, dst = city_names[i], city_names[j]
                # Dijkstra
                dist_map = {c: float("inf") for c in city_names}
                dist_map[src] = 0
                visited = set()

                while len(visited) < len(city_names):
                    unvisited = {c: dist_map[c] for c in city_names if c not in visited}
                    if not unvisited:
                        break
                    curr = min(unvisited, key=unvisited.get)
                    if dist_map[curr] == float("inf"):
                        break
                    visited.add(curr)

                    for nxt, l in adj[curr]:
                        if dist_map[curr] + l < dist_map[nxt]:
                            dist_map[nxt] = dist_map[curr] + l

                shortest_path_len = dist_map[dst]
                if 2 <= shortest_path_len < float("inf"):
                    points = max(2, min(22, int(round(shortest_path_len * 1.2))))
                    valid_ticket_pairs.append((src, dst, points))

        rng.shuffle(valid_ticket_pairs)
        selected_pairs = valid_ticket_pairs[: self.config.num_tickets]

        tickets: list[DestinationTicket] = []
        for idx, (c_a, c_b, points) in enumerate(selected_pairs):
            t_id = f"t_{idx}_{c_a[:4].lower()}_{c_b[:4].lower()}"
            tickets.append(DestinationTicket(id=t_id, city_a=c_a, city_b=c_b, points=points))

        return tickets
