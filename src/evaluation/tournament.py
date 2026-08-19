"""Round-robin and Swiss tournament orchestrator."""

from itertools import combinations
from typing import Any

from src.agents.base_agent import BaseAgent
from src.evaluation.elo import EloSystem
from src.evaluation.evaluator import Evaluator


class Tournament:
    """Orchestrates deterministic round-robin tournaments among agent pools."""

    def __init__(
        self,
        agents: list[BaseAgent],
        games_per_pair: int = 50,
        initial_elo: float = 1200.0,
        k_factor: float = 32.0,
        evaluator: Evaluator | None = None,
        board: Any = None,
        tickets_deck: Any = None,
    ) -> None:
        self.agents = agents
        self.games_per_pair = games_per_pair
        self.elo_system = EloSystem(initial_rating=initial_elo, k_factor=k_factor)
        if evaluator is not None:
            self.evaluator = evaluator
        elif board is not None or tickets_deck is not None:
            self.evaluator = Evaluator(board=board, tickets_deck=tickets_deck)
        else:
            self.evaluator = Evaluator()

    def run(self, seed: int = 42) -> dict[str, Any]:
        """Execute round-robin pairings, update Elo, and compile leaderboard."""
        matchup_results: list[dict[str, Any]] = []
        agent_stats: dict[str, dict[str, Any]] = {
            a.name: {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "total_score": 0.0,
                "total_turns": 0.0,
            }
            for a in self.agents
        }

        pairings = list(combinations(self.agents, 2))
        current_seed = seed

        for agent_a, agent_b in pairings:
            pair_results = self.evaluator.evaluate(
                agent_a=agent_a,
                agent_b=agent_b,
                num_games=self.games_per_pair,
                seed=current_seed,
            )
            current_seed += self.games_per_pair

            metrics_a = pair_results[agent_a.name]
            metrics_b = pair_results[agent_b.name]

            # Update cumulative stats
            for agent, metrics in [(agent_a, metrics_a), (agent_b, metrics_b)]:
                st = agent_stats[agent.name]
                st["games"] += metrics.total_games
                st["wins"] += metrics.wins
                st["losses"] += metrics.losses
                st["draws"] += metrics.draws
                st["total_score"] += metrics.avg_score * metrics.total_games
                st["total_turns"] += metrics.avg_turns * metrics.total_games

            # Update Elo ratings based on match results
            # Score contribution for A = (wins + 0.5 * draws) / total_games
            score_a = (
                (metrics_a.wins + 0.5 * metrics_a.draws) / self.games_per_pair
                if self.games_per_pair > 0
                else 0.5
            )
            self.elo_system.update(agent_a.name, agent_b.name, score_a=score_a)

            matchup_results.append({
                "agent_a": agent_a.name,
                "agent_b": agent_b.name,
                "wins_a": metrics_a.wins,
                "wins_b": metrics_b.wins,
                "draws": metrics_a.draws,
                "avg_score_a": metrics_a.avg_score,
                "avg_score_b": metrics_b.avg_score,
            })

        # Build Leaderboard
        leaderboard = []
        for agent in self.agents:
            st = agent_stats[agent.name]
            total_g = st["games"]
            win_rate = (st["wins"] / total_g) if total_g > 0 else 0.0
            avg_score = (st["total_score"] / total_g) if total_g > 0 else 0.0
            rating = self.elo_system.get_rating(agent.name)

            leaderboard.append({
                "name": agent.name,
                "elo": round(rating, 1),
                "win_rate": round(win_rate, 3),
                "wins": st["wins"],
                "losses": st["losses"],
                "draws": st["draws"],
                "avg_score": round(avg_score, 1),
                "total_games": total_g,
            })

        # Sort leaderboard descending by Elo
        leaderboard.sort(key=lambda x: x["elo"], reverse=True)

        return {
            "leaderboard": leaderboard,
            "ratings": {a.name: self.elo_system.get_rating(a.name) for a in self.agents},
            "matchups": matchup_results,
            "seed": seed,
        }
