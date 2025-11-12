"""Graph loading and manipulation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import networkx as nx
import pandas as pd

from core.routes import RouteTuple


def load_ticket_to_ride_graph(city_locations_file: Path | str, routes_file: Path | str) -> nx.MultiGraph:
    graph = nx.MultiGraph()

    city_locations_path = Path(city_locations_file)
    routes_path = Path(routes_file)

    city_coords = _load_city_locations(city_locations_path)
    for city, coords in city_coords.items():
        graph.add_node(city, pos=coords)

    routes = pd.read_csv(routes_path)
    for _, route in routes.iterrows():
        graph.add_edge(
            route["From"],
            route["To"],
            weight=route["Distance"],
            color=route["Color"],
        )

    return graph


def _load_city_locations(path: Path) -> Dict[str, Iterable[float]]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
