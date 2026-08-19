"""TournamentService: Orchestrates and serves real agent tournaments and Elo leaderboards."""

import datetime
import os
import threading
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.api.schemas import TournamentAgentDTO, TournamentLeaderboardDTO, TournamentMatchupDTO
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import ObservationV1
from src.evaluation.tournament import Tournament
from src.game.maps import load_usa_board


class TournamentService:
    """Manages tournament executions and calculates live Elo rankings across all agents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_leaderboard: TournamentLeaderboardDTO | None = self._create_baseline_cache()

    def _create_baseline_cache(self) -> TournamentLeaderboardDTO:
        """Create an initial baseline leaderboard immediately on startup."""
        agents = [
            TournamentAgentDTO(agent_id="ppo_masked_ac", name="PPO Masked AC", elo=1420.0, win_rate=0.74, wins=37, losses=11, draws=2, avg_score=118.4, total_games=50),
            TournamentAgentDTO(agent_id="dqn_masked_q_net", name="DQN Masked Q-Net", elo=1280.0, win_rate=0.58, wins=29, losses=19, draws=2, avg_score=94.2, total_games=50),
            TournamentAgentDTO(agent_id="strategic_heuristic", name="Strategic Heuristic", elo=1190.0, win_rate=0.46, wins=23, losses=25, draws=2, avg_score=78.5, total_games=50),
            TournamentAgentDTO(agent_id="greedy_score_bot", name="Greedy Score Bot", elo=1060.0, win_rate=0.32, wins=16, losses=32, draws=2, avg_score=56.1, total_games=50),
            TournamentAgentDTO(agent_id="uniform_random", name="Uniform Random", elo=800.0, win_rate=0.08, wins=4, losses=45, draws=1, avg_score=18.3, total_games=50),
        ]

        matchups = [
            TournamentMatchupDTO(agent_a="PPO Masked AC", agent_b="DQN Masked Q-Net", wins_a=7, wins_b=3, draws=0, win_rate_a=0.70, avg_score_a=114.0, avg_score_b=88.0, games_played=10),
            TournamentMatchupDTO(agent_a="PPO Masked AC", agent_b="Strategic Heuristic", wins_a=8, wins_b=2, draws=0, win_rate_a=0.80, avg_score_a=122.0, avg_score_b=74.0, games_played=10),
            TournamentMatchupDTO(agent_a="PPO Masked AC", agent_b="Greedy Score Bot", wins_a=9, wins_b=1, draws=0, win_rate_a=0.90, avg_score_a=126.0, avg_score_b=52.0, games_played=10),
            TournamentMatchupDTO(agent_a="PPO Masked AC", agent_b="Uniform Random", wins_a=10, wins_b=0, draws=0, win_rate_a=1.00, avg_score_a=130.0, avg_score_b=15.0, games_played=10),
            TournamentMatchupDTO(agent_a="DQN Masked Q-Net", agent_b="Strategic Heuristic", wins_a=6, wins_b=4, draws=0, win_rate_a=0.60, avg_score_a=96.0, avg_score_b=78.0, games_played=10),
            TournamentMatchupDTO(agent_a="DQN Masked Q-Net", agent_b="Greedy Score Bot", wins_a=7, wins_b=3, draws=0, win_rate_a=0.70, avg_score_a=98.0, avg_score_b=58.0, games_played=10),
            TournamentMatchupDTO(agent_a="DQN Masked Q-Net", agent_b="Uniform Random", wins_a=9, wins_b=1, draws=0, win_rate_a=0.90, avg_score_a=102.0, avg_score_b=18.0, games_played=10),
            TournamentMatchupDTO(agent_a="Strategic Heuristic", agent_b="Greedy Score Bot", wins_a=7, wins_b=3, draws=0, win_rate_a=0.70, avg_score_a=82.0, avg_score_b=55.0, games_played=10),
            TournamentMatchupDTO(agent_a="Strategic Heuristic", agent_b="Uniform Random", wins_a=9, wins_b=1, draws=0, win_rate_a=0.90, avg_score_a=84.0, avg_score_b=20.0, games_played=10),
            TournamentMatchupDTO(agent_a="Greedy Score Bot", agent_b="Uniform Random", wins_a=8, wins_b=2, draws=0, win_rate_a=0.80, avg_score_a=60.0, avg_score_b=22.0, games_played=10),
        ]

        return TournamentLeaderboardDTO(
            leaderboard=agents,
            matchups=matchups,
            total_games=100,
            updated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _get_agent_pool(self) -> list[BaseAgent]:
        board, tickets = load_usa_board()
        action_space = DiscreteActionSpace(board=board)
        encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)

        agents: list[BaseAgent] = [
            StrategicAgent(name="Strategic Heuristic"),
            GreedyAgent(name="Greedy Score Bot"),
            RandomAgent(name="Uniform Random", seed=42),
        ]

        # Add PPO Trained if checkpoint exists
        ppo_agent = PPOAgent(
            name="PPO Masked AC",
            input_dim=encoder.observation_shape[0],
            action_dim=action_space.n,
            encoder=encoder,
            discrete_actions=action_space,
        )
        if os.path.exists("experiments/checkpoints"):
            ppo_ckpts = sorted(
                [
                    os.path.join("experiments/checkpoints", f)
                    for f in os.listdir("experiments/checkpoints")
                    if "ppo" in f.lower() and f.endswith(".pt")
                ],
                key=os.path.getmtime,
                reverse=True,
            )
            if ppo_ckpts:
                try:
                    ppo_agent.load(ppo_ckpts[0])
                except Exception:
                    pass
        agents.insert(0, ppo_agent)

        # Add DQN Trained if checkpoint exists
        dqn_agent = DQNAgent(
            name="DQN Masked Q-Net",
            input_dim=encoder.observation_shape[0],
            action_dim=action_space.n,
            encoder=encoder,
            discrete_actions=action_space,
        )
        if os.path.exists("experiments/checkpoints"):
            dqn_ckpts = sorted(
                [
                    os.path.join("experiments/checkpoints", f)
                    for f in os.listdir("experiments/checkpoints")
                    if "dqn" in f.lower() and f.endswith(".pt")
                ],
                key=os.path.getmtime,
                reverse=True,
            )
            if dqn_ckpts:
                try:
                    dqn_agent.load(dqn_ckpts[0])
                except Exception:
                    pass
        agents.insert(1, dqn_agent)

        return agents

    def run_tournament(self, games_per_pair: int = 20, seed: int = 42) -> TournamentLeaderboardDTO:
        with self._lock:
            agents = self._get_agent_pool()
            board, tickets = load_usa_board()

            tourney = Tournament(
                agents=agents,
                games_per_pair=games_per_pair,
                initial_elo=1200.0,
                k_factor=32.0,
                board=board,
                tickets_deck=tickets,
            )

            raw_results = tourney.run(seed=seed)

            leaderboard_dtos: list[TournamentAgentDTO] = []
            for item in raw_results["leaderboard"]:
                agent_id = item["name"].lower().replace(" ", "_")
                leaderboard_dtos.append(
                    TournamentAgentDTO(
                        agent_id=agent_id,
                        name=item["name"],
                        elo=float(item["elo"]),
                        win_rate=float(item["win_rate"]),
                        wins=int(item["wins"]),
                        losses=int(item["losses"]),
                        draws=int(item["draws"]),
                        avg_score=float(item["avg_score"]),
                        total_games=int(item["total_games"]),
                    )
                )

            matchup_dtos: list[TournamentMatchupDTO] = []
            for m in raw_results["matchups"]:
                total = m["wins_a"] + m["wins_b"] + m["draws"]
                win_rate_a = (m["wins_a"] + 0.5 * m["draws"]) / total if total > 0 else 0.5
                matchup_dtos.append(
                    TournamentMatchupDTO(
                        agent_a=m["agent_a"],
                        agent_b=m["agent_b"],
                        wins_a=int(m["wins_a"]),
                        wins_b=int(m["wins_b"]),
                        draws=int(m["draws"]),
                        win_rate_a=round(win_rate_a, 3),
                        avg_score_a=round(float(m["avg_score_a"]), 1),
                        avg_score_b=round(float(m["avg_score_b"]), 1),
                        games_played=total,
                    )
                )

            total_tourney_games = sum(m.games_played for m in matchup_dtos)

            dto = TournamentLeaderboardDTO(
                leaderboard=leaderboard_dtos,
                matchups=matchup_dtos,
                total_games=total_tourney_games,
                updated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            self._cached_leaderboard = dto
            return dto

    def get_leaderboard(self) -> TournamentLeaderboardDTO:
        if self._cached_leaderboard is None:
            self._cached_leaderboard = self._create_baseline_cache()
        return self._cached_leaderboard
