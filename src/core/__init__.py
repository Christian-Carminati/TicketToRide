"""Shared core utilities for Ticket to Ride."""

from core.constants import POINTS_TABLE, PointsTable, TOTAL_TRAINS
from core.routes import RouteTuple, extract_routes, unique_routes_from_graph
from core.scoring import ScoreBreakdown, calculate_longest_path, calculate_route_points, calculate_ticket_points, score_routes, trains_used
from core.solution import Solution

__all__ = [
    "POINTS_TABLE",
    "PointsTable",
    "TOTAL_TRAINS",
    "RouteTuple",
    "extract_routes",
    "unique_routes_from_graph",
    "ScoreBreakdown",
    "calculate_longest_path",
    "calculate_route_points",
    "calculate_ticket_points",
    "score_routes",
    "trains_used",
    "Solution",
]
