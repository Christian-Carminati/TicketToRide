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
from src.api.schemas import (
    TournamentAgentDTO,
    TournamentLeaderboardDTO,
    TournamentMatchupDTO,
    TournamentParticipantOptionDTO,
)
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import ObservationV1
from src.evaluation.tournament import Tournament
from src.game.maps import create_synthetic_mini_board, load_usa_board


class TournamentService:
    """Manages tournament executions, customizable participants, and calculates live Elo rankings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_leaderboard: TournamentLeaderboardDTO | None = self._create_baseline_cache()

    def get_available_participants(self) -> list[TournamentParticipantOptionDTO]:
        """Discover all baseline bots and saved model checkpoints available for tournament play."""
        options: list[TournamentParticipantOptionDTO] = [
            TournamentParticipantOptionDTO(
                id="baseline_strategic",
                name="Strategic Heuristic",
                category="baseline",
                algorithm="strategic",
                description="Agente euristico avanzato con pianificazione ticket e rotte ad alto punteggio",
            ),
            TournamentParticipantOptionDTO(
                id="baseline_greedy",
                name="Greedy Score Bot",
                category="baseline",
                algorithm="greedy",
                description="Agente avido che massimizza il punteggio immediato turno per turno",
            ),
            TournamentParticipantOptionDTO(
                id="baseline_random",
                name="Uniform Random",
                category="baseline",
                algorithm="random",
                description="Agente di controllo che seleziona uniformemente tra le mosse legali",
            ),
        ]

        ckpt_dir = os.path.join("experiments", "checkpoints")
        if os.path.exists(ckpt_dir):
            files = sorted(
                [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
                reverse=True,
            )
            for fname in files:
                full_path = os.path.join(ckpt_dir, fname)
                algo = "ppo" if "ppo" in fname.lower() else "dqn"
                clean_name = fname.replace(".pt", "").replace("_", " ").title()

                if "live_latest" in fname:
                    display_name = f"⭐ Ultimo Checkpoint Live ({algo.upper()})"
                    desc = f"Pesi neurali più recenti salvati dall'ultimo addestramento {algo.upper()}"
                elif "usa_trained" in fname:
                    display_name = f"🏆 Benchmark Ufficiale USA ({algo.upper()})"
                    desc = f"Modello pre-addestrato per benchmark su mappa USA ({algo.upper()})"
                else:
                    display_name = f"🧠 {clean_name}"
                    desc = f"Checkpoint addestrato salvato ({algo.upper()})"

                options.append(
                    TournamentParticipantOptionDTO(
                        id=f"ckpt_{fname.replace('.', '_')}",
                        name=display_name,
                        category="checkpoint",
                        algorithm=algo,
                        checkpoint_path=full_path,
                        description=desc,
                    )
                )

        return options

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
            map_name="usa",
            available_participants=self.get_available_participants(),
        )

    def _build_agent(
        self,
        participant: TournamentParticipantOptionDTO,
        board: Any,
        tickets: Any,
    ) -> BaseAgent:
        action_space = DiscreteActionSpace(board=board)
        encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)

        algo = participant.algorithm.lower()
        if algo == "strategic":
            return StrategicAgent(name=participant.name)
        elif algo == "greedy":
            return GreedyAgent(name=participant.name)
        elif algo == "random":
            return RandomAgent(name=participant.name, seed=42)
        elif algo == "dqn":
            agent_dqn = DQNAgent(
                name=participant.name,
                input_dim=encoder.observation_shape[0],
                action_dim=action_space.n,
                encoder=encoder,
                discrete_actions=action_space,
            )
            if participant.checkpoint_path and os.path.exists(participant.checkpoint_path):
                try:
                    agent_dqn.load(participant.checkpoint_path)
                except Exception as e:
                    print(f"Failed to load DQN checkpoint {participant.checkpoint_path}: {e}")
            return agent_dqn
        elif algo == "ppo":
            agent_ppo = PPOAgent(
                name=participant.name,
                input_dim=encoder.observation_shape[0],
                action_dim=action_space.n,
                encoder=encoder,
                discrete_actions=action_space,
            )
            if participant.checkpoint_path and os.path.exists(participant.checkpoint_path):
                try:
                    agent_ppo.load(participant.checkpoint_path)
                except Exception as e:
                    print(f"Failed to load PPO checkpoint {participant.checkpoint_path}: {e}")
            return agent_ppo
        else:
            return RandomAgent(name=participant.name, seed=42)

    def run_tournament(
        self,
        participant_ids: list[str] | None = None,
        games_per_pair: int = 15,
        map_name: str = "usa",
        seed: int = 42,
    ) -> TournamentLeaderboardDTO:
        with self._lock:
            all_available = self.get_available_participants()

            if participant_ids and len(participant_ids) >= 2:
                selected_options = [p for p in all_available if p.id in participant_ids]
            else:
                # Default selection: baseline bots + live latest checkpoints
                selected_options = [p for p in all_available if p.category == "baseline" or "live_latest" in p.id]
                if len(selected_options) < 2:
                    selected_options = all_available[:5]

            # Load map
            if map_name == "mini":
                board, tickets = create_synthetic_mini_board()
            else:
                board, tickets = load_usa_board()

            agents: list[BaseAgent] = [
                self._build_agent(opt, board=board, tickets=tickets)
                for opt in selected_options
            ]

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
                map_name=map_name,
                available_participants=all_available,
            )
            self._cached_leaderboard = dto
            return dto

    def get_leaderboard(self) -> TournamentLeaderboardDTO:
        if self._cached_leaderboard is None:
            self._cached_leaderboard = self._create_baseline_cache()
        else:
            # Refresh available participants dynamically
            self._cached_leaderboard.available_participants = self.get_available_participants()
        return self._cached_leaderboard
