"""Utility functions to evaluate Ticket to Ride board states.

These helpers mirror the scoring logic used in the heuristic and optimal solvers
but are provided here so RL environments and opponents can share a common
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import pandas as pd

RouteTuple = Tuple[str, str, int, str]


@dataclass(frozen=True)
class PointsTable:
    """Points gained for a claimed route of a specific length."""

    one: int = 1
    two: int = 2
    three: int = 4
    four: int = 7
    five: int = 10
    six: int = 15

    def __getitem__(self, length: int) -> int:
        mapping = {
            1: self.one,
            2: self.two,
            3: self.three,
            4: self.four,
            5: self.five,
            6: self.six,
        }
        return mapping.get(length, 0)


POINTS_TABLE = PointsTable()


def calculate_route_points(routes: Sequence[RouteTuple]) -> int:
    """Return the total points obtained from the provided routes."""

    return sum(POINTS_TABLE[length] for _, _, length, _ in routes)


def _build_graph(routes: Sequence[RouteTuple]) -> nx.Graph:
    graph = nx.Graph()
    for city1, city2, length, _ in routes:
        graph.add_edge(city1, city2, weight=length)
    return graph


def calculate_longest_path(routes: Sequence[RouteTuple]) -> int:
    """Compute the length of the longest continuous path for the given routes."""

    if not routes:
        return 0

    graph = _build_graph(routes)
    max_length = 0
    for node in graph.nodes:
        visited_edges = set()
        length = _dfs_longest_path(graph, node, visited_edges, 0)
        max_length = max(max_length, length)
    return max_length


def _dfs_longest_path(
    graph: nx.Graph,
    node: str,
    visited_edges: set[Tuple[str, str]],
    current_length: int,
) -> int:
    max_length = current_length
    for neighbor in graph.neighbors(node):
        edge = tuple(sorted((node, neighbor)))
        if edge in visited_edges:
            continue
        visited_edges.add(edge)
        edge_length = graph[node][neighbor]["weight"]
        path_length = _dfs_longest_path(graph, neighbor, visited_edges, current_length + edge_length)
        max_length = max(max_length, path_length)
        visited_edges.remove(edge)
    return max_length


def calculate_ticket_points(routes: Sequence[RouteTuple], tickets_df: pd.DataFrame) -> int:
    """Return the total ticket points that can be satisfied with current routes."""

    if not routes:
        return 0

    route_graph = nx.Graph()
    for city1, city2, _, _ in routes:
        route_graph.add_edge(city1, city2)

    ticket_points = 0
    for _, row in tickets_df.iterrows():
        city1 = row["From"]
        city2 = row["To"]
        points = int(row["Points"])
        if city1 in route_graph and city2 in route_graph and nx.has_path(route_graph, city1, city2):
            ticket_points += points
    return ticket_points


def final_score(routes: Sequence[RouteTuple], tickets_df: pd.DataFrame) -> int:
    """Compute the final total score for the given set of routes."""

    route_points = calculate_route_points(routes)
    longest = calculate_longest_path(routes)
    longest_bonus = 10 if longest > 0 else 0
    ticket_points = calculate_ticket_points(routes, tickets_df)
    return route_points + longest_bonus + ticket_points


def unique_routes_from_graph(graph: nx.MultiGraph) -> List[RouteTuple]:
    """Extract a list of unique routes from the Board graph.

    For multiple edges between the same cities we keep the longest route since
    it yields the highest reward in the simplified rule-set used by the
    optimisation solvers.
    """

    unique_routes: dict[Tuple[str, str], RouteTuple] = {}
    for u, v, data in graph.edges(data=True):
        length = data.get("weight", 1)
        color = data.get("color", "X")
        pair = tuple(sorted((u, v)))
        if pair not in unique_routes or length > unique_routes[pair][2]:
            unique_routes[pair] = (u, v, length, color)
    return list(unique_routes.values())


def trains_used(routes: Iterable[RouteTuple]) -> int:
    return sum(route[2] for route in routes)
