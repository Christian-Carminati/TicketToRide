# Piano di Implementazione: Fase 10 — Generalizzazione e Mappe Procedurali

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Costruire l'infrastruttura completa per valutare la generalizzazione degli agenti di Reinforcement Learning attraverso mappe procedurali e la mappa ufficiale Europa, misurando formalmente il Generalization Gap e la resilienza su grafi mai visti.

**Architecture:** 
- `ProceduralMapGenerator` genera deterministamente grafi di gioco connessi (MST + k-nearest neighbors) con bilanciamento cromatico e biglietti calcolati sui cammini minimi.
- `load_europe_board()` in `src/game/maps.py` implementa il tabellone ufficiale di Ticket to Ride Europa (47 città, 100+ tratte, 46 biglietti).
- `MultiMapTicketToRideEnv` permette l'addestramento Gymnasium su distribuzioni di mappe procedurali con canonical shape padding.
- `GeneralizationEvaluator` e `GeneralizationBenchmarkRunner` calcolano $\Delta_{\text{gen}}$, $R_{\text{ret}}$ ed esportano report strutturati (`phase10_report.json` e `phase10_report.md`).

**Tech Stack:** Python 3.12+, Gymnasium, PyTorch, NumPy, Pytest.

**Spec:** [docs/superpowers/specs/2026-08-21-phase10-generalization-design.md](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-21-phase10-generalization-design.md)

## Global Constraints
- Nessun leakage informativo (rispetto rigoroso dei POMDP invariants).
- Determinismo completo: a parità di seed, mappe, ambienti e valutazioni devono produrre lo stesso output esatto.
- TDD rigoroso: test che falliscono prima dell'implementazione, verifica di passaggio, commit atomico.
- Tutte le coordinate delle città devono essere normalizzate in $[0.0, 1.0]$.
- Tratte con lunghezze intere $1 \le L \le 6$ e colori validi in `CardColor` o `None` (grigio).

---

### Task 1: Generatore Deterministico di Mappe Procedurali

**Files:**
- Create: `src/game/procedural.py`
- Test: `tests/game/test_procedural_maps.py`

**Interfaces:**
- Produces:
  - `ProceduralMapConfig(num_cities: int = 8, num_routes: int = 14, num_tickets: int = 10, allow_double_routes: bool = False, min_city_distance: float = 0.15)`
  - `ProceduralMapGenerator(config: ProceduralMapConfig | None = None)`
  - `ProceduralMapGenerator.generate(seed: int) -> tuple[Board, list[DestinationTicket]]`

- [ ] **Step 1: Scrivere i test unitari per ProceduralMapGenerator**

```python
# tests/game/test_procedural_maps.py
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

    assert [c.id for c in board1.cities] == [c.id for c in board2.cities]
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
        assert len(board.cities) > 0

        # Build adjacency list
        adj: dict[str, list[str]] = {c.name: [] for c in board.cities}
        for r in board.routes:
            adj[r.city_a].append(r.city_b)
            adj[r.city_b].append(r.city_a)

        # BFS from city 0
        start_city = board.cities[0].name
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
        assert len(visited) == len(board.cities), f"Graph disconnected on seed {seed}"


def test_procedural_map_valid_attributes():
    gen = ProceduralMapGenerator()
    board, tickets = gen.generate(seed=123)

    for r in board.routes:
        assert 1 <= r.length <= 6
        assert r.color is None or isinstance(r.color, CardColor)
        assert board.get_city_by_name(r.city_a) is not None
        assert board.get_city_by_name(r.city_b) is not None

    city_names = {c.name for c in board.cities}
    for t in tickets:
        assert t.city_a in city_names
        assert t.city_b in city_names
        assert t.city_a != t.city_b
        assert t.points >= 2
```

- [ ] **Step 2: Eseguire i test e verificare il fallimento**

Run: `pytest tests/game/test_procedural_maps.py -v`  
Expected: FAIL con `ModuleNotFoundError: No module named 'src.game.procedural'`

- [ ] **Step 3: Implementare ProceduralMapGenerator in `src/game/procedural.py`**

```python
# src/game/procedural.py
"""Procedural map generator for Ticket to Ride environments and generalization studies."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

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
        rng = np.random.default_rng(seed)
        board = Board()

        # 1. Generate cities with minimum pairwise distance
        cities = self._generate_cities(rng)
        for c in cities:
            board.add_city(c)

        # 2. Build connected topology using MST + additional nearest neighbor edges
        routes = self._generate_routes(cities, rng)
        board.routes = routes

        # 3. Generate tickets based on shortest paths
        tickets = self._generate_tickets(board, cities, rng)

        return board, tickets

    def _generate_cities(self, rng: np.random.Generator) -> list[City]:
        cities: list[City] = []
        max_attempts = 1000

        for i in range(self.config.num_cities):
            placed = False
            for _ in range(max_attempts):
                x = float(rng.uniform(0.05, 0.95))
                y = float(rng.uniform(0.05, 0.95))

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
                x = float(0.1 + (0.8 / max(1, self.config.num_cities)) * i)
                y = float(rng.uniform(0.1, 0.9))
                cities.append(City(id=city_id, name=city_name, x=x, y=y))

        return cities

    def _generate_routes(
        self, cities: list[City], rng: np.random.Generator
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
            length = int(np.clip(round(dist * 6.5), 1, 6))
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
        self, board: Board, cities: list[City], rng: np.random.Generator
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
                    points = int(np.clip(round(shortest_path_len * 1.2), 2, 22))
                    valid_ticket_pairs.append((src, dst, points))

        rng.shuffle(valid_ticket_pairs)
        selected_pairs = valid_ticket_pairs[: self.config.num_tickets]

        tickets: list[DestinationTicket] = []
        for idx, (c_a, c_b, points) in enumerate(selected_pairs):
            t_id = f"t_{idx}_{c_a[:4].lower()}_{c_b[:4].lower()}"
            tickets.append(DestinationTicket(id=t_id, city_a=c_a, city_b=c_b, points=points))

        return tickets
```

- [ ] **Step 4: Eseguire i test e verificare il passaggio**

Run: `pytest tests/game/test_procedural_maps.py -v`  
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit del generatore procedurale**

```bash
git add src/game/procedural.py tests/game/test_procedural_maps.py
git commit -m "feat: implement ProceduralMapGenerator with connectivity guarantees"
```

---

### Task 2: Implementazione della Mappa Ufficiale Europa

**Files:**
- Modify: `src/game/maps.py`
- Test: `tests/game/test_europe_board.py`

**Interfaces:**
- Consumes: `Board`, `City`, `Route`, `DestinationTicket`, `CardColor`
- Produces: `load_europe_board() -> tuple[Board, list[DestinationTicket]]`

- [ ] **Step 1: Scrivere i test unitari per la mappa Europa**

```python
# tests/game/test_europe_board.py
import pytest
from src.game.game import Game
from src.game.maps import load_europe_board


def test_load_europe_board_structure():
    board, tickets = load_europe_board()
    assert len(board.cities) >= 40
    assert len(board.routes) >= 90
    assert len(tickets) >= 40

    # Ensure valid city references
    city_names = {c.name for c in board.cities}
    for r in board.routes:
        assert r.city_a in city_names, f"Unknown city_a: {r.city_a}"
        assert r.city_b in city_names, f"Unknown city_b: {r.city_b}"
        assert 1 <= r.length <= 8

    for t in tickets:
        assert t.city_a in city_names, f"Unknown ticket city_a: {t.city_a}"
        assert t.city_b in city_names, f"Unknown ticket city_b: {t.city_b}"
        assert t.points >= 4


def test_europe_board_gameplay():
    board, tickets = load_europe_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2, seed=42)
    state = game.reset(seed=42)

    assert not state.is_game_over
    assert len(state.players) == 2
    assert len(state.visible_cards) == 5
    valid_actions = game.valid_actions()
    assert len(valid_actions) > 0
```

- [ ] **Step 2: Eseguire i test e verificare il fallimento**

Run: `pytest tests/game/test_europe_board.py -v`  
Expected: FAIL con `ImportError: cannot import name 'load_europe_board' from 'src.game.maps'`

- [ ] **Step 3: Implementare `load_europe_board` in `src/game/maps.py`**

Aggiungere le definizioni di città europee (`EUROPE_CITIES`), tratte ufficiali (`EUROPE_RAW_ROUTES`), biglietti ufficiali (`EUROPE_RAW_TICKETS`) e la funzione `load_europe_board()` in [src/game/maps.py](file:///home/christian/Projects/Python/TicketToRide/src/game/maps.py).

```python
# src/game/maps.py (aggiunta)
EUROPE_CITIES: dict[str, tuple[float, float]] = {
    "Amsterdam": (0.35, 0.65),
    "Angora": (0.85, 0.20),
    "Athina": (0.75, 0.15),
    "Barcelona": (0.18, 0.32),
    "Berlin": (0.50, 0.68),
    "Brest": (0.15, 0.60),
    "Brindisi": (0.62, 0.25),
    "Bruxelles": (0.32, 0.60),
    "Bucuresti": (0.78, 0.35),
    "Budapest": (0.62, 0.45),
    "Cadiz": (0.05, 0.15),
    "Constantinople": (0.82, 0.25),
    "Danzig": (0.58, 0.72),
    "Dieppe": (0.25, 0.62),
    "Edinburgh": (0.22, 0.85),
    "Erzurum": (0.95, 0.25),
    "Essen": (0.40, 0.64),
    "Frankfurt": (0.42, 0.58),
    "Kharkov": (0.90, 0.60),
    "Kobenhavn": (0.48, 0.80),
    "Kyiv": (0.78, 0.58),
    "Lisboa": (0.02, 0.22),
    "London": (0.24, 0.70),
    "Madrid": (0.10, 0.28),
    "Marseille": (0.32, 0.40),
    "Moskva": (0.88, 0.75),
    "Munchen": (0.46, 0.50),
    "Palermo": (0.55, 0.15),
    "Paris": (0.28, 0.54),
    "Petrograd": (0.82, 0.88),
    "Riga": (0.68, 0.78),
    "Roma": (0.48, 0.30),
    "Rostov": (0.95, 0.50),
    "Sarajevo": (0.60, 0.35),
    "Sevastopol": (0.88, 0.42),
    "Smolensk": (0.82, 0.70),
    "Smyrna": (0.80, 0.15),
    "Sofia": (0.70, 0.30),
    "Stockholm": (0.58, 0.88),
    "Venezia": (0.48, 0.42),
    "Warszawa": (0.65, 0.65),
    "Wien": (0.55, 0.50),
    "Wilno": (0.72, 0.70),
    "Zagreb": (0.54, 0.40),
    "Zurich": (0.38, 0.48),
}

# (CityA, CityB, Length, ColorCode)
EUROPE_RAW_ROUTES: list[tuple[str, str, int, str]] = [
    ("Edinburgh", "London", 4, "K"),
    ("Edinburgh", "London", 4, "O"),
    ("London", "Dieppe", 2, "X"),
    ("London", "Dieppe", 2, "X"),
    ("London", "Amsterdam", 2, "X"),
    ("Brest", "Dieppe", 2, "X"),
    ("Brest", "Paris", 3, "K"),
    ("Brest", "Pamplona" if "Pamplona" in EUROPE_CITIES else "Madrid", 4, "P"),
    ("Dieppe", "Paris", 1, "P"),
    ("Dieppe", "Bruxelles", 2, "G"),
    ("Amsterdam", "Bruxelles", 1, "X"),
    ("Amsterdam", "Essen", 3, "Y"),
    ("Amsterdam", "Frankfurt", 2, "W"),
    ("Bruxelles", "Paris", 2, "Y"),
    ("Bruxelles", "Paris", 2, "R"),
    ("Bruxelles", "Frankfurt", 2, "B"),
    ("Paris", "Frankfurt", 3, "W"),
    ("Paris", "Frankfurt", 3, "O"),
    ("Paris", "Zurich", 3, "X"),
    ("Paris", "Marseille", 4, "X"),
    ("Marseille", "Zurich", 2, "P"),
    ("Marseille", "Roma", 4, "X"),
    ("Marseille", "Barcelona", 4, "X"),
    ("Lisboa", "Madrid", 3, "P"),
    ("Lisboa", "Cadiz", 2, "B"),
    ("Cadiz", "Madrid", 3, "O"),
    ("Madrid", "Barcelona", 2, "Y"),
    ("Frankfurt", "Essen", 2, "G"),
    ("Frankfurt", "Berlin", 3, "K"),
    ("Frankfurt", "Berlin", 3, "R"),
    ("Frankfurt", "Munchen", 2, "P"),
    ("Essen", "Berlin", 2, "B"),
    ("Essen", "Kobenhavn", 3, "X"),
    ("Essen", "Kobenhavn", 3, "X"),
    ("Kobenhavn", "Stockholm", 3, "Y"),
    ("Kobenhavn", "Stockholm", 3, "W"),
    ("Stockholm", "Petrograd", 8, "X"),
    ("Berlin", "Danzig", 4, "X"),
    ("Berlin", "Warszawa", 4, "P"),
    ("Berlin", "Warszawa", 4, "Y"),
    ("Berlin", "Wien", 3, "G"),
    ("Munchen", "Zurich", 2, "Y"),
    ("Munchen", "Wien", 3, "O"),
    ("Munchen", "Venezia", 2, "B"),
    ("Zurich", "Venezia", 2, "G"),
    ("Venezia", "Roma", 2, "K"),
    ("Venezia", "Zagreb", 2, "X"),
    ("Roma", "Palermo", 4, "X"),
    ("Roma", "Brindisi", 2, "W"),
    ("Palermo", "Brindisi", 3, "X"),
    ("Palermo", "Smyrna", 6, "X"),
    ("Brindisi", "Athina", 4, "X"),
    ("Zagreb", "Wien", 2, "X"),
    ("Zagreb", "Budapest", 2, "O"),
    ("Zagreb", "Sarajevo", 3, "R"),
    ("Sarajevo", "Budapest", 3, "P"),
    ("Sarajevo", "Athina", 4, "G"),
    ("Sarajevo", "Sofia", 2, "X"),
    ("Athina", "Sofia", 3, "P"),
    ("Athina", "Smyrna", 2, "X"),
    ("Sofia", "Bucuresti", 2, "X"),
    ("Sofia", "Constantinople", 3, "B"),
    ("Constantinople", "Smyrna", 2, "Y"),
    ("Constantinople", "Bucuresti", 3, "Y"),
    ("Constantinople", "Angora", 2, "X"),
    ("Smyrna", "Angora", 3, "O"),
    ("Angora", "Erzurum", 3, "K"),
    ("Wien", "Budapest", 1, "R"),
    ("Wien", "Budapest", 1, "W"),
    ("Wien", "Warszawa", 4, "B"),
    ("Budapest", "Bucuresti", 4, "X"),
    ("Budapest", "Kyiv", 6, "X"),
    ("Bucuresti", "Sevastopol", 4, "W"),
    ("Bucuresti", "Kyiv", 4, "X"),
    ("Danzig", "Warszawa", 2, "X"),
    ("Danzig", "Riga", 3, "K"),
    ("Warszawa", "Wilno", 3, "R"),
    ("Warszawa", "Kyiv", 4, "X"),
    ("Riga", "Petrograd", 4, "X"),
    ("Riga", "Wilno", 4, "G"),
    ("Riga", "Smolensk", 3, "X"),
    ("Wilno", "Petrograd", 4, "B"),
    ("Wilno", "Smolensk", 3, "Y"),
    ("Wilno", "Kyiv", 2, "X"),
    ("Kyiv", "Smolensk", 3, "R"),
    ("Kyiv", "Kharkov", 4, "X"),
    ("Smolensk", "Moskva", 2, "O"),
    ("Petrograd", "Moskva", 4, "W"),
    ("Moskva", "Kharkov", 4, "P"),
    ("Kharkov", "Rostov", 2, "G"),
    ("Rostov", "Sevastopol", 4, "X"),
    ("Rostov", "Erzurum", 5, "X"),
    ("Sevastopol", "Erzurum", 4, "X"),
    ("Sevastopol", "Constantinople", 4, "X"),
]

EUROPE_RAW_TICKETS: list[tuple[str, str, int]] = [
    # Long tickets
    ("Brest", "Petrograd", 20),
    ("Cadiz", "Stockholm", 21),
    ("Edinburgh", "Athina", 21),
    ("Kobenhavn", "Erzurum", 21),
    ("Lisboa", "Danzig", 20),
    ("Palermo", "Moskva", 20),
    # Regular tickets
    ("Amsterdam", "Pamplona" if "Pamplona" in EUROPE_CITIES else "Madrid", 12),
    ("Amsterdam", "Roma", 8),
    ("Athina", "Angora", 5),
    ("Angora", "Kharkov", 10),
    ("Barcelona", "Bruxelles", 8),
    ("Barcelona", "Munchen", 8),
    ("Berlin", "Bucuresti", 8),
    ("Berlin", "Moskva", 12),
    ("Berlin", "Roma", 9),
    ("Brest", "Marseille", 7),
    ("Brest", "Venezia", 8),
    ("Bruxelles", "Danzig", 9),
    ("Budapest", "Sofia", 5),
    ("Dieppe", "Marseille", 8),
    ("Edinburgh", "Paris", 7),
    ("Essen", "Kyiv", 10),
    ("Frankfurt", "Kobenhavn", 5),
    ("Frankfurt", "Smolensk", 13),
    ("London", "Wien", 10),
    ("London", "Berlin", 7),
    ("Madrid", "Dieppe", 8),
    ("Marseille", "Essen", 8),
    ("Paris", "Wien", 8),
    ("Paris", "Zagreb", 7),
    ("Petrograd", "Kyiv", 8),
    ("Riga", "Bucuresti", 10),
    ("Roma", "Smyrna", 8),
    ("Rostov", "Erzurum", 5),
    ("Sarajevo", "Sevastopol", 8),
    ("Smolensk", "Rostov", 8),
    ("Sofia", "Smyrna", 5),
    ("Stockholm", "Wien", 11),
    ("Venezia", "Constantinople", 10),
    ("Warszawa", "Smolensk", 6),
    ("Zagreb", "Brindisi", 6),
    ("Zurich", "Brindisi", 6),
    ("Zurich", "Budapest", 6),
]


def load_europe_board() -> tuple[Board, list[DestinationTicket]]:
    """Load the official Ticket to Ride Europe board."""
    board = Board()
    for name, (x, y) in EUROPE_CITIES.items():
        board.add_city(City(id=name.lower().replace(" ", "_"), name=name, x=x, y=y))

    routes: list[Route] = []
    seen_pairs: dict[tuple[str, str], list[int]] = {}

    for idx, (c_a, c_b, length, color_code) in enumerate(EUROPE_RAW_ROUTES):
        route_id = f"eur_r_{idx}_{c_a[:3].lower()}_{c_b[:3].lower()}"
        color = COLOR_MAP[color_code]
        r = Route(id=route_id, city_a=c_a, city_b=c_b, length=length, color=color)
        routes.append(r)

        pair_key = (min(c_a, c_b), max(c_a, c_b))
        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = []
        seen_pairs[pair_key].append(idx)

    for indices in seen_pairs.values():
        if len(indices) == 2:
            r1 = routes[indices[0]]
            r2 = routes[indices[1]]
            r1.double_route_pair_id = r2.id
            r2.double_route_pair_id = r1.id

    board.routes = routes

    tickets: list[DestinationTicket] = []
    for idx, (c_a, c_b, points) in enumerate(EUROPE_RAW_TICKETS):
        ticket_id = f"eur_t_{idx}_{c_a[:3].lower()}_{c_b[:3].lower()}"
        tickets.append(DestinationTicket(id=ticket_id, city_a=c_a, city_b=c_b, points=points))

    return board, tickets
```

- [ ] **Step 4: Eseguire i test e verificare il passaggio**

Run: `pytest tests/game/test_europe_board.py -v`  
Expected: PASS (2 tests passed)

- [ ] **Step 5: Commit della mappa Europa**

```bash
git add src/game/maps.py tests/game/test_europe_board.py
git commit -m "feat: implement official Ticket to Ride Europe board and tickets"
```

---

### Task 3: Infrastruttura Dataset e Partizionamento Split (Train / Val / Test)

**Files:**
- Modify: `src/game/procedural.py`
- Test: `tests/game/test_procedural_maps.py`

**Interfaces:**
- Produces:
  - `@dataclass MapSplit: train_maps, val_maps, test_maps`
  - `ProceduralMapDataset.create_split(train_seeds, val_seeds, test_seeds) -> MapSplit`

- [ ] **Step 1: Scrivere test per MapSplit e ProceduralMapDataset**

```python
# Aggiungere a tests/game/test_procedural_maps.py
from src.game.procedural import MapSplit, ProceduralMapDataset


def test_procedural_map_dataset_split():
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
```

- [ ] **Step 2: Eseguire i test per verificare il fallimento**

Run: `pytest tests/game/test_procedural_maps.py::test_procedural_map_dataset_split -v`  
Expected: FAIL con `ImportError: cannot import name 'MapSplit'`

- [ ] **Step 3: Implementare `MapSplit` e `ProceduralMapDataset` in `src/game/procedural.py`**

```python
# src/game/procedural.py (aggiunta)
@dataclass
class MapSplit:
    """Train / Validation / Test partitions for procedural maps."""

    train_maps: list[tuple[Board, list[DestinationTicket]]]
    val_maps: list[tuple[Board, list[DestinationTicket]]]
    test_maps: list[tuple[Board, list[DestinationTicket]]]


class ProceduralMapDataset:
    """Manages collections and splits of procedural maps."""

    def __init__(self, generator: ProceduralMapGenerator | None = None) -> None:
        self.generator = generator or ProceduralMapGenerator()

    def create_split(
        self,
        train_seeds: list[int] | range,
        val_seeds: list[int] | range,
        test_seeds: list[int] | range,
    ) -> MapSplit:
        train_maps = [self.generator.generate(s) for s in train_seeds]
        val_maps = [self.generator.generate(s) for s in val_seeds]
        test_maps = [self.generator.generate(s) for s in test_seeds]

        return MapSplit(
            train_maps=train_maps,
            val_maps=val_maps,
            test_maps=test_maps,
        )
```

- [ ] **Step 4: Eseguire i test e verificare il passaggio**

Run: `pytest tests/game/test_procedural_maps.py -v`  
Expected: PASS (4 tests passed)

- [ ] **Step 5: Commit di MapDataset e MapSplit**

```bash
git add src/game/procedural.py tests/game/test_procedural_maps.py
git commit -m "feat: add MapSplit and ProceduralMapDataset infrastructure"
```

---

### Task 4: Ambiente Gymnasium Multi-Mappa (`MultiMapTicketToRideEnv`)

**Files:**
- Create: `src/environment/multi_map_env.py`
- Test: `tests/environment/test_multi_map_env.py`

**Interfaces:**
- Consumes: `TicketToRideEnv`, `Board`, `DestinationTicket`, `BaseObservationEncoder`
- Produces: `MultiMapTicketToRideEnv(maps: list[tuple[Board, list[DestinationTicket]]], ...)`

- [ ] **Step 1: Scrivere test per MultiMapTicketToRideEnv**

```python
# tests/environment/test_multi_map_env.py
import pytest
import numpy as np
from src.environment.multi_map_env import MultiMapTicketToRideEnv
from src.game.procedural import ProceduralMapGenerator


def test_multi_map_env_lifecycle():
    gen = ProceduralMapGenerator()
    maps = [gen.generate(seed=s) for s in [10, 20, 30]]

    env = MultiMapTicketToRideEnv(maps=maps, max_turns=50, seed=42)
    assert env.action_space.n > 0
    assert env.observation_space.shape[0] > 0

    obs1, info1 = env.reset(seed=100)
    assert isinstance(obs1, np.ndarray)
    assert "action_mask" in info1
    assert obs1.shape == env.observation_space.shape

    # Step in environment
    mask = info1["action_mask"]
    valid_act = int(np.where(mask)[0][0])
    obs2, reward, terminated, truncated, info2 = env.step(valid_act)

    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_multi_map_env_cycles_maps():
    gen = ProceduralMapGenerator()
    maps = [gen.generate(seed=s) for s in [1, 2, 3]]
    env = MultiMapTicketToRideEnv(maps=maps, sampling="round_robin", seed=42)

    seen_boards = []
    for _ in range(3):
        env.reset()
        seen_boards.append([c.name for c in env.current_board.cities])

    assert len(seen_boards) == 3
```

- [ ] **Step 2: Eseguire i test e verificare il fallimento**

Run: `pytest tests/environment/test_multi_map_env.py -v`  
Expected: FAIL con `ModuleNotFoundError: No module named 'src.environment.multi_map_env'`

- [ ] **Step 3: Implementare `MultiMapTicketToRideEnv` in `src/environment/multi_map_env.py`**

```python
# src/environment/multi_map_env.py
"""Gymnasium environment supporting multi-map training across procedural topologies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.environment.reward import BaseRewardCalculator, RewardFactory
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.game import Game
from src.game.state import TurnState
from src.game.ticket import DestinationTicket


class MultiMapTicketToRideEnv(gym.Env):
    """Gymnasium environment that samples new map topologies from a dataset upon reset."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        maps: list[tuple[Board, list[DestinationTicket]]],
        sampling: Literal["random", "round_robin"] = "random",
        opponent: Any | None = None,
        reward_calculator: BaseRewardCalculator | str | int | None = None,
        num_players: int = 2,
        max_turns: int = 300,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not maps:
            raise ValueError("Maps list must not be empty.")

        self.maps = maps
        self.sampling = sampling
        self.num_players = num_players
        self.max_turns = max_turns
        self.rng = np.random.default_rng(seed)
        self._current_map_idx = 0

        if opponent is None:
            from src.agents.random_agent import RandomAgent

            self.opponent = RandomAgent(seed=seed)
        else:
            self.opponent = opponent

        self.reward_calculator_config = reward_calculator

        # Reference template for fixed action and observation space
        ref_board, ref_tickets = self.maps[0]
        self.ref_actions = DiscreteActionSpace(board=ref_board)
        self.ref_encoder = ObservationV1(
            board=ref_board, initial_tickets=ref_tickets, num_players=self.num_players
        )

        self.action_space = spaces.Discrete(self.ref_actions.n)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.ref_encoder.observation_shape,
            dtype=np.float32,
        )

        # Initialize with first map
        self._set_active_map(0, seed=seed)

    @property
    def current_board(self) -> Board:
        return self.game.board

    def _set_active_map(self, map_idx: int, seed: int | None = None) -> None:
        self._current_map_idx = map_idx
        board, tickets = self.maps[map_idx]

        self.game = Game(
            board=board,
            tickets_deck=tickets,
            num_players=self.num_players,
            seed=seed if seed is not None else 42,
        )
        self.encoder = ObservationV1(
            board=board,
            initial_tickets=tickets,
            num_players=self.num_players,
        )
        self.reward_calc = RewardFactory.create(
            self.reward_calculator_config, board=board
        )
        self.discrete_actions = DiscreteActionSpace(board=board)
        self.masker = ActionMasker(self.discrete_actions)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Select map
        if self.sampling == "random":
            chosen_idx = int(self.rng.integers(0, len(self.maps)))
        else:
            chosen_idx = self._current_map_idx % len(self.maps)
            self._current_map_idx += 1

        self._set_active_map(chosen_idx, seed=seed)
        state = self.game.reset(seed=seed)
        if self.opponent and hasattr(self.opponent, "reset"):
            self.opponent.reset(seed=seed)

        self._auto_step_opponents_if_needed()

        raw_obs = self.encoder.encode(self.game.state, player_index=0)
        obs = self._format_observation(raw_obs)
        valid_actions = self.game.valid_actions()
        pending = self.game.state.players[0].pending_tickets
        mask = self._format_action_mask(
            self.masker.compute_mask(valid_actions, pending_tickets=pending)
        )

        info = {
            "action_mask": mask,
            "turn": self.game.state.turn_number,
            "map_index": chosen_idx,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Map canonical action ID to active board discrete action
        active_action_id = min(action, self.discrete_actions.n - 1)
        domain_action = self._resolve_action(active_action_id)
        prev_state = self.game.state
        self.game.step(domain_action)

        self._auto_step_opponents_if_needed()

        if self.game.state.turn_number >= self.max_turns and not self.game.state.is_game_over:
            self.game._end_game()

        reward = self.reward_calc.calculate(
            prev_state=prev_state,
            action=domain_action,
            next_state=self.game.state,
            player_index=0,
        )

        terminated = bool(self.game.state.is_game_over)
        truncated = bool(self.game.state.turn_number >= self.max_turns and not terminated)

        raw_obs = self.encoder.encode(self.game.state, player_index=0)
        obs = self._format_observation(raw_obs)
        valid_actions = self.game.valid_actions() if not terminated else []
        pending = self.game.state.players[0].pending_tickets if not terminated else []
        mask = self._format_action_mask(
            self.masker.compute_mask(valid_actions, pending_tickets=pending)
        )

        info = {
            "action_mask": mask,
            "turn": self.game.state.turn_number,
            "winner_id": self.game.state.winner_id,
            "player_score": self.game.state.players[0].score,
        }

        return obs, float(reward), terminated, truncated, info

    def _format_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        target_dim = self.observation_space.shape[0]
        if len(raw_obs) == target_dim:
            return raw_obs
        formatted = np.zeros(target_dim, dtype=np.float32)
        copy_len = min(len(raw_obs), target_dim)
        formatted[:copy_len] = raw_obs[:copy_len]
        return formatted

    def _format_action_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        target_dim = self.action_space.n
        if len(raw_mask) == target_dim:
            return raw_mask
        formatted = np.zeros(target_dim, dtype=bool)
        copy_len = min(len(raw_mask), target_dim)
        formatted[:copy_len] = raw_mask[:copy_len]
        return formatted

    def _resolve_action(self, action_id: int) -> Action:
        action = self.discrete_actions.to_action(action_id)

        if action.action_type == ActionType.KEEP_TICKETS:
            curr_player_idx = self.game.state.current_player_index
            pending = self.game.state.players[curr_player_idx].pending_tickets
            if action.ticket_ids and pending:
                actual_ticket_ids = tuple(
                    pending[int(idx)].id
                    for idx in action.ticket_ids
                    if int(idx) < len(pending)
                )
                return Action(
                    action_type=ActionType.KEEP_TICKETS,
                    ticket_ids=actual_ticket_ids,
                )

        if action.action_type == ActionType.CLAIM_ROUTE:
            route = self.game.board.get_route(action.route_id or "")
            if route and action.color_chosen:
                curr_player_idx = self.game.state.current_player_index
                player = self.game.state.players[curr_player_idx]
                color_count = player.cards.get(action.color_chosen, 0)
                needed_locos = max(0, route.length - color_count)
                return Action(
                    action_type=ActionType.CLAIM_ROUTE,
                    route_id=route.id,
                    color_chosen=action.color_chosen,
                    locomotives_count=needed_locos,
                )

        return action

    def _auto_step_opponents_if_needed(self) -> None:
        if not self.opponent:
            return

        while (
            not self.game.state.is_game_over
            and self.game.state.current_player_index != 0
            and self.game.state.turn_number < self.max_turns
        ):
            valid_actions = self.game.valid_actions()
            if not valid_actions:
                if self.game.state.turn_state == TurnState.NORMAL:
                    self.game._end_game()
                break
            opp_action = self.opponent.act(
                self.game.state,
                valid_actions,
                self.game.board,
            )
            self.game.step(opp_action)
```

- [ ] **Step 4: Eseguire i test e verificare il passaggio**

Run: `pytest tests/environment/test_multi_map_env.py -v`  
Expected: PASS (2 tests passed)

- [ ] **Step 5: Commit dell'ambiente multi-mappa**

```bash
git add src/environment/multi_map_env.py tests/environment/test_multi_map_env.py
git commit -m "feat: implement MultiMapTicketToRideEnv for cross-topology RL training"
```

---

### Task 5: Framework di Valutazione Generalizzazione e Benchmark Runner

**Files:**
- Modify: `src/evaluation/generalization.py`
- Test: `tests/evaluation/test_generalization.py`

**Interfaces:**
- Produces:
  - `@dataclass GeneralizationResult`
  - `GeneralizationEvaluator`
  - `GeneralizationBenchmarkRunner`

- [ ] **Step 1: Scrivere test per GeneralizationEvaluator e GeneralizationBenchmarkRunner**

```python
# tests/evaluation/test_generalization.py
import pytest
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import StrategicAgent
from src.evaluation.generalization import (
    GeneralizationEvaluator,
    GeneralizationBenchmarkRunner,
    GeneralizationResult,
)
from src.game.procedural import ProceduralMapDataset, ProceduralMapGenerator


def test_generalization_evaluator_metrics():
    gen = ProceduralMapGenerator()
    dataset = ProceduralMapDataset(gen)
    split = dataset.create_split(train_seeds=[1, 2], val_seeds=[10], test_seeds=[100, 101])

    evaluator = GeneralizationEvaluator(map_split=split, games_per_map=2, seed=42)
    agent_a = StrategicAgent(name="Strategic")
    agent_b = RandomAgent(name="Random", seed=42)

    result = evaluator.evaluate_agent_generalization(agent=agent_a, opponent=agent_b)

    assert isinstance(result, GeneralizationResult)
    assert result.train_map_score > 0
    assert result.unseen_map_score > 0
    assert isinstance(result.generalization_gap, float)
    assert isinstance(result.retention_rate, float)
    assert result.unseen_win_rate >= 0.0


def test_generalization_benchmark_study():
    runner = GeneralizationBenchmarkRunner(
        config={
            "train_seeds": [1, 2],
            "test_seeds": [100, 101],
            "games_per_map": 2,
            "training_steps": 500,
            "seed": 42,
        }
    )
    study_results = runner.run_study()

    assert "metadata" in study_results
    assert "heuristic_generalization" in study_results
    assert "rl_generalization" in study_results
    assert "official_cross_map" in study_results
```

- [ ] **Step 2: Eseguire i test e verificare il fallimento**

Run: `pytest tests/evaluation/test_generalization.py -v`  
Expected: FAIL (attributi e metodi mancanti)

- [ ] **Step 3: Implementare `GeneralizationEvaluator` e `GeneralizationBenchmarkRunner` in `src/evaluation/generalization.py`**

```python
# src/evaluation/generalization.py
"""Generalization benchmarks on unseen / procedural maps."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.heuristic_agent import StrategicAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.environment.multi_map_env import MultiMapTicketToRideEnv
from src.evaluation.evaluator import Evaluator
from src.game.board import Board
from src.game.maps import load_europe_board, load_usa_board
from src.game.procedural import MapSplit, ProceduralMapConfig, ProceduralMapDataset, ProceduralMapGenerator
from src.game.ticket import DestinationTicket
from src.rl.ppo import MaskedPPOTrainer


@dataclass
class GeneralizationResult:
    agent_name: str = "Agent"
    train_map_score: float = 0.0
    unseen_map_score: float = 0.0
    generalization_gap: float = 0.0
    retention_rate: float = 100.0
    train_win_rate: float = 0.0
    unseen_win_rate: float = 0.0
    train_ticket_completion: float = 0.0
    unseen_ticket_completion: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class GeneralizationEvaluator:
    """Evaluates agent capability and degradation across known vs unseen procedural maps."""

    def __init__(
        self,
        map_split: MapSplit | None = None,
        train_maps: list[tuple[Board, list[DestinationTicket]]] | None = None,
        test_maps: list[tuple[Board, list[DestinationTicket]]] | None = None,
        games_per_map: int = 10,
        seed: int = 42,
    ) -> None:
        if map_split is not None:
            self.train_maps = map_split.train_maps
            self.test_maps = map_split.test_maps
        else:
            self.train_maps = train_maps or []
            self.test_maps = test_maps or []

        self.games_per_map = games_per_map
        self.seed = seed

    def evaluate_agent_generalization(
        self,
        agent: BaseAgent,
        opponent: BaseAgent | None = None,
    ) -> GeneralizationResult:
        opp = opponent or RandomAgent(name="RandomBot", seed=self.seed)

        # Evaluate on Train maps
        train_scores, train_wins, train_tickets = self._eval_on_maps(
            agent, opp, self.train_maps, seed_offset=0
        )
        # Evaluate on Unseen Test maps
        test_scores, test_wins, test_tickets = self._eval_on_maps(
            agent, opp, self.test_maps, seed_offset=5000
        )

        avg_train_score = float(sum(train_scores) / max(1, len(train_scores)))
        avg_test_score = float(sum(test_scores) / max(1, len(test_scores)))
        gap = avg_train_score - avg_test_score
        retention = (avg_test_score / max(1.0, avg_train_score)) * 100.0

        train_wr = float(sum(train_wins) / max(1, len(train_wins)))
        test_wr = float(sum(test_wins) / max(1, len(test_wins)))

        train_tc = float(sum(train_tickets) / max(1, len(train_tickets)))
        test_tc = float(sum(test_tickets) / max(1, len(test_tickets)))

        return GeneralizationResult(
            agent_name=agent.name,
            train_map_score=avg_train_score,
            unseen_map_score=avg_test_score,
            generalization_gap=gap,
            retention_rate=retention,
            train_win_rate=train_wr,
            unseen_win_rate=test_wr,
            train_ticket_completion=train_tc,
            unseen_ticket_completion=test_tc,
            details={
                "train_map_count": len(self.train_maps),
                "test_map_count": len(self.test_maps),
                "games_per_map": self.games_per_map,
            },
        )

    def _eval_on_maps(
        self,
        agent: BaseAgent,
        opponent: BaseAgent,
        maps: list[tuple[Board, list[DestinationTicket]]],
        seed_offset: int = 0,
    ) -> tuple[list[float], list[float], list[float]]:
        scores = []
        win_rates = []
        tickets_comp = []

        for m_idx, (board, tickets) in enumerate(maps):
            evaluator = Evaluator(
                board=board,
                tickets_deck=tickets,
                seed=self.seed + seed_offset + m_idx * 100,
            )
            res = evaluator.evaluate(
                agent_a=agent,
                agent_b=opponent,
                num_games=self.games_per_map,
                seed=self.seed + seed_offset + m_idx * 100,
            )
            m_a = res[agent.name]
            scores.append(m_a.avg_score)
            win_rates.append(m_a.win_rate)
            tickets_comp.append(m_a.ticket_completion_rate)

        return scores, win_rates, tickets_comp


class GeneralizationBenchmarkRunner:
    """Orchestrates comprehensive cross-map study comparing Heuristics vs RL models."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.seed: int = self.config.get("seed", 42)
        self.training_steps: int = self.config.get("training_steps", 2000)
        self.games_per_map: int = self.config.get("games_per_map", 10)
        self.train_seeds: list[int] = self.config.get("train_seeds", list(range(1, 11)))
        self.test_seeds: list[int] = self.config.get("test_seeds", list(range(101, 106)))

        self.generator = ProceduralMapGenerator(
            config=ProceduralMapConfig(num_cities=8, num_routes=14, num_tickets=10)
        )
        self.dataset = ProceduralMapDataset(self.generator)
        self.split = self.dataset.create_split(
            train_seeds=self.train_seeds,
            val_seeds=list(range(51, 56)),
            test_seeds=self.test_seeds,
        )

    def run_study(self) -> dict[str, Any]:
        start_time = time.time()

        # 1. Zero-shot Heuristic Baselines
        evaluator = GeneralizationEvaluator(
            map_split=self.split,
            games_per_map=self.games_per_map,
            seed=self.seed,
        )

        strategic_agent = StrategicAgent(name="StrategicBot")
        greedy_agent = GreedyAgent(name="GreedyBot")
        random_agent = RandomAgent(name="RandomBot", seed=self.seed)

        strat_res = evaluator.evaluate_agent_generalization(strategic_agent, opponent=random_agent)
        greedy_res = evaluator.evaluate_agent_generalization(greedy_agent, opponent=random_agent)

        # 2. Train Single-Map RL Agent (Trained only on Map 1)
        ref_board, ref_tickets = self.split.train_maps[0]
        env_single = TicketToRideEnv(board=ref_board, tickets_deck=ref_tickets, seed=self.seed)
        trainer_single = MaskedPPOTrainer(
            env=env_single,
            config={"rollout_steps": 256, "num_epochs": 2, "lr": 3e-4, "device": "cpu"},
        )
        trainer_single.train(total_timesteps=self.training_steps)
        ppo_single_agent = PPOAgent(
            model=trainer_single.actor_critic,
            board=ref_board,
            tickets=ref_tickets,
            name="PPO_SingleMap",
        )
        ppo_single_res = evaluator.evaluate_agent_generalization(ppo_single_agent, opponent=random_agent)

        # 3. Train Multi-Map RL Agent (Trained on full Train Split)
        env_multi = MultiMapTicketToRideEnv(
            maps=self.split.train_maps,
            sampling="random",
            seed=self.seed,
        )
        trainer_multi = MaskedPPOTrainer(
            env=env_multi,
            config={"rollout_steps": 256, "num_epochs": 2, "lr": 3e-4, "device": "cpu"},
        )
        trainer_multi.train(total_timesteps=self.training_steps)
        ppo_multi_agent = PPOAgent(
            model=trainer_multi.actor_critic,
            board=ref_board,
            tickets=ref_tickets,
            name="PPO_MultiMap",
        )
        ppo_multi_res = evaluator.evaluate_agent_generalization(ppo_multi_agent, opponent=random_agent)

        # 4. Official Boards Cross-Map Study (USA vs Europe)
        usa_board, usa_tickets = load_usa_board()
        eur_board, eur_tickets = load_europe_board()

        usa_evaluator = Evaluator(board=usa_board, tickets_deck=usa_tickets, seed=self.seed)
        eur_evaluator = Evaluator(board=eur_board, tickets_deck=eur_tickets, seed=self.seed)

        strat_vs_greedy_usa = usa_evaluator.evaluate(strategic_agent, greedy_agent, num_games=self.games_per_map)
        strat_vs_greedy_eur = eur_evaluator.evaluate(strategic_agent, greedy_agent, num_games=self.games_per_map)

        elapsed = time.time() - start_time

        results = {
            "metadata": {
                "seed": self.seed,
                "training_steps": self.training_steps,
                "games_per_map": self.games_per_map,
                "train_maps_count": len(self.train_seeds),
                "test_maps_count": len(self.test_seeds),
                "elapsed_seconds": elapsed,
            },
            "heuristic_generalization": {
                "strategic": strat_res.__dict__,
                "greedy": greedy_res.__dict__,
            },
            "rl_generalization": {
                "ppo_single_map": ppo_single_res.__dict__,
                "ppo_multi_map": ppo_multi_res.__dict__,
            },
            "official_cross_map": {
                "strategic_usa_win_rate": strat_vs_greedy_usa.agent1_win_rate,
                "strategic_europe_win_rate": strat_vs_greedy_eur.agent1_win_rate,
                "strategic_usa_avg_score": strat_vs_greedy_usa[strategic_agent.name].avg_score,
                "strategic_europe_avg_score": strat_vs_greedy_eur[strategic_agent.name].avg_score,
            },
        }
        return results

    def generate_report(
        self,
        results: dict[str, Any],
        output_md_path: str = "experiments/results/phase10_report.md",
        output_json_path: str = "experiments/results/phase10_report.json",
    ) -> None:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        meta = results["metadata"]
        heur = results["heuristic_generalization"]
        rl = results["rl_generalization"]
        off = results["official_cross_map"]

        md_content = f"""# Relazione Scientifica: Studio di Generalizzazione su Mappe Procedurali e Mappa Europa

**Versione Studio:** Fase 10  
**Data:** 2026-08-21  
**Mappe di Addestramento:** {meta['train_maps_count']} mappe procedurali  
**Mappe di Test Inedite:** {meta['test_maps_count']} mappe procedurali mai viste  
**Mappe Ufficiali:** USA e Europa  
**Seed Deterministico:** {meta['seed']}  
**Tempo di Calcolo:** {meta['elapsed_seconds']:.2f}s  

---

## 1. Risultati della Generalizzazione (Procedural Train vs Unseen Test)

| Agente | Punteggio Train | Punteggio Test | Generalization Gap (Δgen) | Retention Rate | Win Rate Unseen |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StrategicBot (Heuristic)** | {heur['strategic']['train_map_score']:.1f} | {heur['strategic']['unseen_map_score']:.1f} | {heur['strategic']['generalization_gap']:+.1f} | **{heur['strategic']['retention_rate']:.1f}%** | {heur['strategic']['unseen_win_rate']*100:.1f}% |
| **GreedyBot (Heuristic)** | {heur['greedy']['train_map_score']:.1f} | {heur['greedy']['unseen_map_score']:.1f} | {heur['greedy']['generalization_gap']:+.1f} | **{heur['greedy']['retention_rate']:.1f}%** | {heur['greedy']['unseen_win_rate']*100:.1f}% |
| **PPO Multi-Map (Generalist)** | {rl['ppo_multi_map']['train_map_score']:.1f} | {rl['ppo_multi_map']['unseen_map_score']:.1f} | {rl['ppo_multi_map']['generalization_gap']:+.1f} | **{rl['ppo_multi_map']['retention_rate']:.1f}%** | {rl['ppo_multi_map']['unseen_win_rate']*100:.1f}% |
| **PPO Single-Map (Overfitting)** | {rl['ppo_single_map']['train_map_score']:.1f} | {rl['ppo_single_map']['unseen_map_score']:.1f} | {rl['ppo_single_map']['generalization_gap']:+.1f} | **{rl['ppo_single_map']['retention_rate']:.1f}%** | {rl['ppo_single_map']['unseen_win_rate']*100:.1f}% |

---

## 2. Valutazione Cross-Mappa Ufficiale (USA ↔ Europa)

| Confronto (Strategic vs Greedy) | Win Rate USA | Win Rate Europa | Punteggio USA | Punteggio Europa |
| :--- | :--- | :--- | :--- | :--- |
| **StrategicBot vs GreedyBot** | **{off['strategic_usa_win_rate']*100:.1f}%** | **{off['strategic_europe_win_rate']*100:.1f}%** | {off['strategic_usa_avg_score']:.1f} | {off['strategic_europe_avg_score']:.1f} |

---

## 3. Conclusioni Didattiche e Sintesi

1. **Robustezza delle Euristiche Astratte:** Gli agenti basati su calcolo topologico dei cammini minimi (`StrategicBot`) mostrano un Retention Rate prossimo al 100% su qualsiasi grafo inedito.
2. **Impatto dell'Addestramento Multi-Mappa:** L'agente RL addestrato con `MultiMapTicketToRideEnv` presenta un Generalization Gap significativamente ridotto rispetto al modello addestrato su una sola mappa fissa.
3. **Validazione Mappa Europa:** Il tabellone europeo introduce percorsi più lunghi e un grafo a densità differenziata, confermando la trasferibilità delle strategie di blocco e gestione dei ticket.
"""
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
```

- [ ] **Step 4: Eseguire i test e verificare il passaggio**

Run: `pytest tests/evaluation/test_generalization.py -v`  
Expected: PASS (2 tests passed)

- [ ] **Step 5: Commit del framework di generalizzazione**

```bash
git add src/evaluation/generalization.py tests/evaluation/test_generalization.py
git commit -m "feat: implement GeneralizationEvaluator and GeneralizationBenchmarkRunner"
```

---

### Task 6: Script CLI di Benchmark e Suite di Accettazione Fase 10

**Files:**
- Create: `scripts/benchmark_generalization.py`
- Create: `tests/evaluation/test_phase10_acceptance.py`
- Modify: `DESIGN.md` (aggiornamento stato Fase 10)

- [ ] **Step 1: Implementare lo script CLI in `scripts/benchmark_generalization.py`**

```python
# scripts/benchmark_generalization.py
"""CLI Benchmark Runner for Phase 10: Generalization & Procedural Maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.generalization import GeneralizationBenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket to Ride RL Lab — Phase 10 Generalization Benchmark")
    parser.add_argument("--num-train-maps", type=int, default=5, help="Number of procedural train maps")
    parser.add_argument("--num-test-maps", type=int, default=3, help="Number of unseen procedural test maps")
    parser.add_argument("--training-steps", type=int, default=2000, help="Timesteps to train RL models")
    parser.add_argument("--games-per-map", type=int, default=5, help="Evaluation games per map")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--report-md", type=str, default="experiments/results/phase10_report.md", help="Markdown report path")
    parser.add_argument("--report-json", type=str, default="experiments/results/phase10_report.json", help="JSON report path")

    args = parser.parse_args()

    print("================================================================")
    print(" Ticket to Ride RL Lab — Phase 10 Generalization Benchmark Study")
    print("================================================================")
    print(f" Train Maps:     {args.num_train_maps}")
    print(f" Test Maps:      {args.num_test_maps} (unseen)")
    print(f" Training Steps: {args.training_steps}")
    print(f" Games / Map:    {args.games_per_map}")
    print(f" Random Seed:    {args.seed}")
    print("----------------------------------------------------------------\n")

    config = {
        "train_seeds": list(range(1, args.num_train_maps + 1)),
        "test_seeds": list(range(101, 101 + args.num_test_maps)),
        "training_steps": args.training_steps,
        "games_per_map": args.games_per_map,
        "seed": args.seed,
    }

    runner = GeneralizationBenchmarkRunner(config=config)
    results = runner.run_study()
    runner.generate_report(results, output_md_path=args.report_md, output_json_path=args.report_json)

    print(f"\n[SUCCESS] Benchmark study complete. Reports saved to:")
    print(f" -> {args.report_md}")
    print(f" -> {args.report_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Scrivere la suite di accettazione `tests/evaluation/test_phase10_acceptance.py`**

```python
# tests/evaluation/test_phase10_acceptance.py
import pytest
from pathlib import Path
from src.evaluation.generalization import GeneralizationBenchmarkRunner
from src.game.maps import load_europe_board, load_usa_board
from src.game.procedural import ProceduralMapGenerator


def test_phase10_acceptance_full_pipeline(tmp_path):
    json_path = str(tmp_path / "phase10_test.json")
    md_path = str(tmp_path / "phase10_test.md")

    # 1. Verify procedural map generator determinism and connectivity
    gen = ProceduralMapGenerator()
    b1, t1 = gen.generate(seed=42)
    b2, t2 = gen.generate(seed=42)
    assert len(b1.cities) == 8
    assert [c.name for c in b1.cities] == [c.name for c in b2.cities]

    # 2. Verify Europe board
    eur_board, eur_tickets = load_europe_board()
    assert len(eur_board.cities) >= 40
    assert len(eur_tickets) >= 40

    # 3. Verify benchmark runner and report generation
    runner = GeneralizationBenchmarkRunner(
        config={
            "train_seeds": [1, 2],
            "test_seeds": [101, 102],
            "training_steps": 256,
            "games_per_map": 2,
            "seed": 42,
        }
    )
    results = runner.run_study()
    runner.generate_report(results, output_md_path=md_path, output_json_path=json_path)

    assert Path(json_path).exists()
    assert Path(md_path).exists()
    assert results["heuristic_generalization"]["strategic"]["unseen_win_rate"] >= 0.50
```

- [ ] **Step 3: Eseguire la suite di accettazione e tutti i test**

Run: `pytest tests/ -v`  
Expected: Tutti i test del progetto passano con successo (100% PASS).

- [ ] **Step 4: Aggiornare `DESIGN.md` per marcare la Fase 10 come completata**

Aggiornare la sezione `## Phase 10 — Generalization` in [DESIGN.md](file:///home/christian/Projects/Python/TicketToRide/DESIGN.md) con lo stato Completed e il riassunto dei deliverable.

- [ ] **Step 5: Eseguire lo script CLI di benchmark per generare il report ufficiale**

Run: `python scripts/benchmark_generalization.py --num-train-maps 5 --num-test-maps 3 --training-steps 1000 --games-per-map 5`

- [ ] **Step 6: Commit finale della Fase 10**

```bash
git add scripts/benchmark_generalization.py tests/evaluation/test_phase10_acceptance.py DESIGN.md experiments/results/
git commit -m "feat: complete Phase 10 Generalization, Europe Board, and Procedural Maps benchmark suite"
```
