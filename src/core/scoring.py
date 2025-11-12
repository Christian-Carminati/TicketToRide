"""
Scoring utilities shared across algorithms and environments.

This module provides functions for calculating scores in Ticket to Ride,
including route points, longest path bonus, and destination ticket points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import networkx as nx
import pandas as pd

from core.constants import POINTS_TABLE
from core.routes import RouteTuple, trains_used

# Bonus points awarded for having the longest continuous path
LONGEST_PATH_BONUS: int = 10


@dataclass(frozen=True)
class ScoreBreakdown:
    """
    Breakdown of a player's score into components.
    
    Attributes:
        route_points: Points from claimed routes (based on length)
        longest_path_length: Length of the longest continuous path
        longest_bonus: Bonus points (10) if longest path > 0
        ticket_points: Points from completed destination tickets
    """

    route_points: int
    longest_path_length: int
    longest_bonus: int
    ticket_points: int

    @property
    def total_score(self) -> int:
        """
        Calculate total score from all components.
        
        Returns:
            Sum of route_points, longest_bonus, and ticket_points
        """
        return self.route_points + self.longest_bonus + self.ticket_points


def calculate_route_points(routes: Sequence[RouteTuple]) -> int:
    return sum(POINTS_TABLE[route[2]] for route in routes)


def _build_weighted_graph(routes: Sequence[RouteTuple]) -> nx.Graph:
    graph = nx.Graph()
    for city1, city2, length, _ in routes:
        graph.add_edge(city1, city2, weight=length)
    return graph


def calculate_longest_path(routes: Sequence[RouteTuple]) -> int:
    if not routes:
        return 0

    graph = _build_weighted_graph(routes)
    max_length = 0
    for node in graph.nodes:
        visited_edges: set[tuple[str, str]] = set()
        length = _dfs_longest_path(graph, node, visited_edges, 0)
        max_length = max(max_length, length)
    return max_length


def _dfs_longest_path(
    graph: nx.Graph,
    node: str,
    visited_edges: set[tuple[str, str]],
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


def score_routes(routes: Sequence[RouteTuple], tickets_df: pd.DataFrame) -> ScoreBreakdown:
    """
    Calculate complete score breakdown for a set of routes.
    
    Args:
        routes: Sequence of routes claimed by the player
        tickets_df: DataFrame containing destination tickets
        
    Returns:
        ScoreBreakdown with all score components
    """
    route_points = calculate_route_points(routes)
    longest = calculate_longest_path(routes)
    longest_bonus = LONGEST_PATH_BONUS if longest > 0 else 0
    ticket_points = calculate_ticket_points(routes, tickets_df)
    return ScoreBreakdown(
        route_points=route_points,
        longest_path_length=longest,
        longest_bonus=longest_bonus,
        ticket_points=ticket_points,
    )


def final_score(routes: Sequence[RouteTuple], tickets_df: pd.DataFrame) -> int:
    return score_routes(routes, tickets_df).total_score


def trains_cost(routes: Iterable[RouteTuple]) -> int:
    return trains_used(routes)
