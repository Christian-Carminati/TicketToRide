"""
Route utilities shared across solvers and environments.

This module provides functions for working with routes in the Ticket to Ride game,
including extraction from graphs, validation, and train usage calculations.

A RouteTuple represents a route as: (city1, city2, length, color)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx

from core.constants import TOTAL_TRAINS

# Type alias for route representation: (city1, city2, length, color)
RouteTuple = Tuple[str, str, int, str]


def extract_routes(graph: nx.MultiGraph) -> List[RouteTuple]:
    """
    Extract all routes from a NetworkX MultiGraph.
    
    Args:
        graph: NetworkX MultiGraph representing the game map
        
    Returns:
        List of all routes as RouteTuple (city1, city2, length, color)
    """
    routes: List[RouteTuple] = []
    for u, v, data in graph.edges(data=True):
        length = data.get("weight", 1)
        color = data.get("color", "X")
        routes.append((u, v, length, color))
    return routes


def unique_routes_from_graph(graph: nx.MultiGraph) -> List[RouteTuple]:
    """
    Extract unique routes from graph, keeping only the longest route between each city pair.
    
    In Ticket to Ride, multiple routes can exist between the same cities (different colors).
    This function keeps only the longest route for each city pair, which is useful for
    solvers that don't need to consider color constraints.
    
    Args:
        graph: NetworkX MultiGraph representing the game map
        
    Returns:
        List of unique routes, one per city pair (the longest)
    """
    unique_routes: Dict[Tuple[str, str], RouteTuple] = {}
    for u, v, data in graph.edges(data=True):
        length = data.get("weight", 1)
        color = data.get("color", "X")
        pair = tuple(sorted((u, v)))
        if pair not in unique_routes or length > unique_routes[pair][2]:
            unique_routes[pair] = (u, v, length, color)
    return list(unique_routes.values())


def trains_used(routes: Iterable[RouteTuple]) -> int:
    """
    Calculate total number of trains used by a collection of routes.
    
    Args:
        routes: Iterable of RouteTuple objects
        
    Returns:
        Total number of trains (sum of route lengths)
    """
    return sum(route[2] for route in routes)


def can_claim_route(
    route: RouteTuple, 
    claimed_pairs: Dict[Tuple[str, str], str], 
    trains_remaining: int
) -> bool:
    """
    Check if a route can be claimed given current constraints.
    
    A route can be claimed if:
    1. The city pair hasn't been claimed yet
    2. The player has enough trains remaining
    
    Args:
        route: Route to check
        claimed_pairs: Dictionary of already claimed city pairs
        trains_remaining: Number of trains the player has left
        
    Returns:
        True if the route can be claimed, False otherwise
    """
    pair = tuple(sorted((route[0], route[1])))
    if pair in claimed_pairs:
        return False
    return trains_remaining >= route[2]
