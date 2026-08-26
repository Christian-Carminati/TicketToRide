"""Round-robin and Swiss tournament orchestrator with multi-core parallel execution."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import os
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
        self.board = board
        self.tickets_deck = tickets_deck
        if evaluator is not None:
            self.evaluator = evaluator
        elif board is not None or tickets_deck is not None:
            self.evaluator = Evaluator(board=board, tickets_deck=tickets_deck)
        else:
            self.evaluator = Evaluator()

    def run(
        self,
        seed: int = 42,
        progress_callback: Any = None,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Execute round-robin pairings, update Elo, and compile leaderboard."""
        if parallel and len(self.agents) > 2:
            return self.run_parallel(seed=seed, progress_callback=progress_callback, max_workers=max_workers)

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
        total_pairings = len(pairings)
        current_seed = seed

        for pair_idx, (agent_a, agent_b) in enumerate(pairings):
            if progress_callback:
                progress_callback({
                    "current_match": pair_idx + 1,
                    "total_matches": total_pairings,
                    "agent_a": agent_a.name,
                    "agent_b": agent_b.name,
                    "status": "running",
                    "percentage": round((pair_idx / max(1, total_pairings)) * 100, 1),
                })

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

            # Update Elo ratings
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

            if progress_callback:
                progress_callback({
                    "current_match": pair_idx + 1,
                    "total_matches": total_pairings,
                    "agent_a": agent_a.name,
                    "agent_b": agent_b.name,
                    "status": "completed",
                    "wins_a": metrics_a.wins,
                    "wins_b": metrics_b.wins,
                    "draws": metrics_a.draws,
                    "percentage": round(((pair_idx + 1) / max(1, total_pairings)) * 100, 1),
                })

        return self._compile_leaderboard(agent_stats, matchup_results, seed)

    def run_parallel(
        self,
        seed: int = 42,
        progress_callback: Any = None,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Execute matchups in parallel across worker threads."""
        pairings = list(combinations(self.agents, 2))
        total_pairings = len(pairings)
        workers = max_workers or min(os.cpu_count() or 4, 16)

        def _evaluate_pair(idx: int, a: BaseAgent, b: BaseAgent) -> tuple[int, BaseAgent, BaseAgent, dict[str, Any]]:
            evaluator = Evaluator(board=self.board, tickets_deck=self.tickets_deck)
            res = evaluator.evaluate(a, b, num_games=self.games_per_pair, seed=seed + idx * 1000)
            return idx, a, b, res

        completed_results: list[tuple[int, BaseAgent, BaseAgent, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_evaluate_pair, idx, a, b): idx
                for idx, (a, b) in enumerate(pairings)
            }
            for future in as_completed(future_to_idx):
                idx, a, b, res = future.result()
                completed_results.append((idx, a, b, res))
                if progress_callback:
                    progress_callback({
                        "current_match": len(completed_results),
                        "total_matches": total_pairings,
                        "agent_a": a.name,
                        "agent_b": b.name,
                        "status": "completed",
                        "percentage": round((len(completed_results) / max(1, total_pairings)) * 100, 1),
                    })

        # Sort back to deterministic pairing order
        completed_results.sort(key=lambda x: x[0])

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
        matchup_results: list[dict[str, Any]] = []

        for _, agent_a, agent_b, pair_results in completed_results:
            metrics_a = pair_results[agent_a.name]
            metrics_b = pair_results[agent_b.name]

            for agent, metrics in [(agent_a, metrics_a), (agent_b, metrics_b)]:
                st = agent_stats[agent.name]
                st["games"] += metrics.total_games
                st["wins"] += metrics.wins
                st["losses"] += metrics.losses
                st["draws"] += metrics.draws
                st["total_score"] += metrics.avg_score * metrics.total_games
                st["total_turns"] += metrics.avg_turns * metrics.total_games

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

        return self._compile_leaderboard(agent_stats, matchup_results, seed)

    def _compile_leaderboard(
        self,
        agent_stats: dict[str, dict[str, Any]],
        matchup_results: list[dict[str, Any]],
        seed: int,
    ) -> dict[str, Any]:
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

        leaderboard.sort(key=lambda x: x["elo"], reverse=True)
        return {
            "leaderboard": leaderboard,
            "ratings": {a.name: self.elo_system.get_rating(a.name) for a in self.agents},
            "matchups": matchup_results,
            "seed": seed,
        }
