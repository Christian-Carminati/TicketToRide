"""Deterministic greedy heuristic agent (alias for backwards compatibility)."""

from src.agents.greedy_agent import GreedyAgent
from src.agents.strategic_agent import StrategicAgent

HeuristicAgent = GreedyAgent

__all__ = ["HeuristicAgent", "StrategicAgent", "GreedyAgent"]
