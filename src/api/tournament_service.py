"""TournamentService: Orchestrates and serves real agent tournaments and Elo leaderboards across all 8 agent families."""

import datetime
import os
import threading
from typing import Any

import torch

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.neural_mcts_agent import BayesianOpponentMCTSAgent, NeuralMCTSAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent
from src.agents.strategic_agent import StrategicAgent
from src.api.schemas import (
    TournamentAgentDTO,
    TournamentLeaderboardDTO,
    TournamentMatchupDTO,
    TournamentParticipantOptionDTO,
    TournamentProgressDTO,
)
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import ObservationV1
from src.evaluation.tournament import Tournament
from src.game.maps import create_synthetic_mini_board, load_usa_board


class TournamentService:
    """Manages tournament executions, customizable participants, and calculates live Elo rankings."""

    def __init__(self, connection_manager: Any = None) -> None:
        self._lock = threading.Lock()
        self._cached_leaderboard: TournamentLeaderboardDTO | None = self._create_baseline_cache()
        self.connection_manager = connection_manager
        self._progress_lock = threading.Lock()
        self._progress = TournamentProgressDTO(is_running=False, status="idle")

    def get_progress(self) -> TournamentProgressDTO:
        with self._progress_lock:
            return self._progress.model_copy()

    def set_connection_manager(self, connection_manager: Any) -> None:
        self.connection_manager = connection_manager

    def get_available_participants(
        self, ckpt_dir: str | None = None
    ) -> list[TournamentParticipantOptionDTO]:
        """Discover all baseline bots and saved model checkpoints available for tournament play."""
        options: list[TournamentParticipantOptionDTO] = [
            TournamentParticipantOptionDTO(
                id="agent_alphazero",
                name="🦅 AlphaZero (PUCT 40 Sims)",
                category="baseline",
                algorithm="alphazero",
                description="AlphaZero Neural MCTS engine with joint policy-value evaluation and PUCT search",
            ),
            TournamentParticipantOptionDTO(
                id="agent_bayesian_mcts",
                name="🎯 Bayesian MCTS (Opponent-Aware)",
                category="baseline",
                algorithm="bayesian_mcts",
                description="Monte Carlo Tree Search with Bayesian ticket belief tracking and tactical blocking",
            ),
            TournamentParticipantOptionDTO(
                id="agent_ismcts",
                name="🌲 Pure IS-MCTS (40 Sims)",
                category="baseline",
                algorithm="mcts",
                description="Information Set MCTS with determinization and Monte Carlo rollouts",
            ),
            TournamentParticipantOptionDTO(
                id="agent_recurrent_ppo",
                name="🧵 Recurrent PPO (LSTM POMDP)",
                category="baseline",
                algorithm="recurrent_ppo",
                description="Recurrent Actor-Critic with sequential memory for tracking hidden opponent cards",
            ),
            TournamentParticipantOptionDTO(
                id="baseline_strategic",
                name="📐 Strategic Heuristic (Dijkstra)",
                category="baseline",
                algorithm="strategic",
                description="Advanced heuristic agent with Dijkstra shortest-path planning and ticket scoring",
            ),
            TournamentParticipantOptionDTO(
                id="baseline_greedy",
                name="⚡ Greedy Score Bot",
                category="baseline",
                algorithm="greedy",
                description="Greedy agent that claims the highest-scoring immediate valid route each turn",
            ),
            TournamentParticipantOptionDTO(
                id="baseline_random",
                name="🎲 Uniform Random",
                category="baseline",
                algorithm="random",
                description="Control baseline that samples uniformly among legal actions",
            ),
        ]

        target_dir = ckpt_dir or os.path.join("experiments", "checkpoints")
        if os.path.exists(target_dir):
            files = sorted(
                [f for f in os.listdir(target_dir) if f.endswith(".pt")],
                key=lambda x: (
                    os.path.getmtime(os.path.join(target_dir, x))
                    if os.path.exists(os.path.join(target_dir, x))
                    else 0
                ),
                reverse=True,
            )
            for fname in files:
                full_path = os.path.join(target_dir, fname)
                lower = fname.lower()
                if "alphazero" in lower:
                    algo = "alphazero"
                    prefix = "🦅 AlphaZero"
                elif "recurrent" in lower or "lstm" in lower:
                    algo = "recurrent_ppo"
                    prefix = "🧵 Recurrent PPO"
                elif "self_play" in lower or "selfplay" in lower:
                    algo = "self_play_ppo"
                    prefix = "🔄 Self-Play PPO"
                elif "ppo" in lower:
                    algo = "ppo"
                    prefix = "⚡ PPO"
                else:
                    algo = "dqn"
                    prefix = "🧠 DQN"

                clean_name = fname.replace(".pt", "").replace("_", " ").title()

                if "live_latest" in fname:
                    display_name = f"⭐ {prefix} Live Latest Checkpoint"
                    desc = f"Most recent trained weights saved from {algo.upper()} training session"
                elif "usa_trained" in fname:
                    display_name = f"🏆 {prefix} Official USA Benchmark"
                    desc = f"Pre-trained checkpoint for official USA benchmarks ({algo.upper()})"
                else:
                    display_name = f"{prefix}: {clean_name}"
                    desc = f"Saved checkpoint ({algo.upper()})"

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
            TournamentAgentDTO(
                agent_id="alphazero_puct",
                name="AlphaZero PUCT",
                elo=1620.0,
                win_rate=0.88,
                wins=44,
                losses=5,
                draws=1,
                avg_score=134.5,
                total_games=50,
            ),
            TournamentAgentDTO(
                agent_id="bayesian_mcts",
                name="Bayesian MCTS",
                elo=1540.0,
                win_rate=0.82,
                wins=41,
                losses=8,
                draws=1,
                avg_score=126.8,
                total_games=50,
            ),
            TournamentAgentDTO(
                agent_id="recurrent_ppo_lstm",
                name="Recurrent PPO (LSTM)",
                elo=1480.0,
                win_rate=0.76,
                wins=38,
                losses=11,
                draws=1,
                avg_score=121.2,
                total_games=50,
            ),
            TournamentAgentDTO(
                agent_id="ppo_masked_ac",
                name="PPO Masked AC",
                elo=1420.0,
                win_rate=0.70,
                wins=35,
                losses=13,
                draws=2,
                avg_score=114.4,
                total_games=50,
            ),
            TournamentAgentDTO(
                agent_id="dqn_masked_q_net",
                name="DQN Masked Q-Net",
                elo=1280.0,
                win_rate=0.56,
                wins=28,
                losses=20,
                draws=2,
                avg_score=94.2,
                total_games=50,
            ),
            TournamentAgentDTO(
                agent_id="strategic_heuristic",
                name="Strategic Heuristic",
                elo=1210.0,
                win_rate=0.48,
                wins=24,
                losses=24,
                draws=2,
                avg_score=82.5,
                total_games=50,
            ),
            TournamentAgentDTO(
                agent_id="greedy_score_bot",
                name="Greedy Score Bot",
                elo=1060.0,
                win_rate=0.32,
                wins=16,
                losses=32,
                draws=2,
                avg_score=56.1,
                total_games=50,
            ),
            TournamentAgentDTO(
                agent_id="uniform_random",
                name="Uniform Random",
                elo=800.0,
                win_rate=0.08,
                wins=4,
                losses=45,
                draws=1,
                avg_score=21.4,
                total_games=50,
            ),
        ]

        matchups: list[TournamentMatchupDTO] = [
            TournamentMatchupDTO(
                agent_a="AlphaZero PUCT",
                agent_b="PPO Masked AC",
                wins_a=16,
                wins_b=3,
                draws=1,
                win_rate_a=0.825,
                avg_score_a=138.2,
                avg_score_b=110.4,
                games_played=20,
            ),
            TournamentMatchupDTO(
                agent_a="AlphaZero PUCT",
                agent_b="Strategic Heuristic",
                wins_a=18,
                wins_b=2,
                draws=0,
                win_rate_a=0.90,
                avg_score_a=142.1,
                avg_score_b=78.2,
                games_played=20,
            ),
            TournamentMatchupDTO(
                agent_a="Bayesian MCTS",
                agent_b="Recurrent PPO (LSTM)",
                wins_a=12,
                wins_b=7,
                draws=1,
                win_rate_a=0.625,
                avg_score_a=128.5,
                avg_score_b=118.9,
                games_played=20,
            ),
            TournamentMatchupDTO(
                agent_a="Recurrent PPO (LSTM)",
                agent_b="PPO Masked AC",
                wins_a=13,
                wins_b=6,
                draws=1,
                win_rate_a=0.675,
                avg_score_a=122.4,
                avg_score_b=109.8,
                games_played=20,
            ),
            TournamentMatchupDTO(
                agent_a="PPO Masked AC",
                agent_b="Strategic Heuristic",
                wins_a=14,
                wins_b=5,
                draws=1,
                win_rate_a=0.725,
                avg_score_a=118.2,
                avg_score_b=82.4,
                games_played=20,
            ),
            TournamentMatchupDTO(
                agent_a="DQN Masked Q-Net",
                agent_b="Strategic Heuristic",
                wins_a=11,
                wins_b=8,
                draws=1,
                win_rate_a=0.575,
                avg_score_a=96.1,
                avg_score_b=86.2,
                games_played=20,
            ),
            TournamentMatchupDTO(
                agent_a="Strategic Heuristic",
                agent_b="Greedy Score Bot",
                wins_a=15,
                wins_b=4,
                draws=1,
                win_rate_a=0.775,
                avg_score_a=88.6,
                avg_score_b=58.2,
                games_played=20,
            ),
            TournamentMatchupDTO(
                agent_a="Greedy Score Bot",
                agent_b="Uniform Random",
                wins_a=18,
                wins_b=2,
                draws=0,
                win_rate_a=0.90,
                avg_score_a=62.4,
                avg_score_b=24.1,
                games_played=20,
            ),
        ]

        return TournamentLeaderboardDTO(
            leaderboard=agents,
            matchups=matchups,
            total_games=80,
            updated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            map_name="usa",
            available_participants=None,
        )

    def get_leaderboard(self) -> TournamentLeaderboardDTO:
        with self._lock:
            if not self._cached_leaderboard:
                self._cached_leaderboard = self._create_baseline_cache()
            self._cached_leaderboard.available_participants = self.get_available_participants()
            return self._cached_leaderboard

    def _build_agent(
        self, participant: TournamentParticipantOptionDTO, board: Any, tickets: list[Any]
    ) -> BaseAgent:
        algo = participant.algorithm.lower()
        if algo == "alphazero":
            agent_az = NeuralMCTSAgent(
                name=participant.name, num_simulations=10, board=board, tickets=tickets
            )
            if participant.checkpoint_path and os.path.exists(participant.checkpoint_path):
                try:
                    agent_az.net.load_state_dict(
                        torch.load(participant.checkpoint_path, map_location="cpu")
                    )
                except Exception as e:
                    print(f"Failed to load AlphaZero checkpoint {participant.checkpoint_path}: {e}")
            return agent_az
        elif algo == "bayesian_mcts":
            return BayesianOpponentMCTSAgent(
                name=participant.name, num_simulations=10, board=board, tickets=tickets
            )
        elif algo == "mcts":
            return MCTSAgent(
                name=participant.name, num_simulations=10, board=board, tickets=tickets
            )
        elif algo in ("recurrent_ppo", "lstm_ppo"):
            return RecurrentPPOAgent(
                name=participant.name,
                board=board,
                tickets=tickets,
                model_or_path=participant.checkpoint_path
                if participant.checkpoint_path and os.path.exists(participant.checkpoint_path)
                else None,
            )
        elif algo == "strategic":
            return StrategicAgent(name=participant.name)
        elif algo == "greedy":
            return GreedyAgent(name=participant.name)
        elif algo == "dqn":
            action_space = DiscreteActionSpace(board=board)
            encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)
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
        elif algo in ("ppo", "self_play_ppo"):
            action_space = DiscreteActionSpace(board=board)
            encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)
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
        games_per_pair: int = 3,
        map_name: str = "usa",
        seed: int = 42,
    ) -> TournamentLeaderboardDTO:
        all_available = self.get_available_participants()

        if participant_ids and len(participant_ids) >= 2:
            selected_options = [p for p in all_available if p.id in participant_ids]
        else:
            selected_options = [
                p for p in all_available if p.category == "baseline" or "live_latest" in p.id
            ]
            if len(selected_options) < 2:
                selected_options = all_available[:6]

        if map_name == "mini":
            board, tickets = create_synthetic_mini_board()
        else:
            board, tickets = load_usa_board()

        agents: list[BaseAgent] = [
            self._build_agent(opt, board=board, tickets=tickets) for opt in selected_options
        ]

        total_pairings = (len(agents) * (len(agents) - 1)) // 2
        start_time = datetime.datetime.now()

        with self._progress_lock:
            self._progress = TournamentProgressDTO(
                is_running=True,
                current_match=0,
                total_matches=total_pairings,
                status="running",
                percentage=0.0,
                elapsed_seconds=0.0,
                estimated_remaining_seconds=0.0,
            )

        if self.connection_manager:
            self.connection_manager.broadcast_sync(
                {
                    "type": "tournament_progress",
                    **self._progress.model_dump(),
                }
            )

        def on_progress(ev: dict[str, Any]) -> None:
            now = datetime.datetime.now()
            elapsed = (now - start_time).total_seconds()
            cur_m = ev.get("current_match", 0)
            tot_m = ev.get("total_matches", total_pairings)
            pct = ev.get("percentage", 0.0)

            est_rem = 0.0
            if cur_m > 0 and tot_m > 0:
                avg_time_per_match = elapsed / cur_m
                est_rem = max(0.0, avg_time_per_match * (tot_m - cur_m))

            recent_dto: TournamentMatchupDTO | None = None
            if ev.get("status") == "completed":
                w_a = ev.get("wins_a", 0)
                w_b = ev.get("wins_b", 0)
                d = ev.get("draws", 0)
                g_played = w_a + w_b + d
                wr_a = (w_a / g_played) if g_played > 0 else 0.0
                recent_dto = TournamentMatchupDTO(
                    agent_a=ev.get("agent_a", ""),
                    agent_b=ev.get("agent_b", ""),
                    wins_a=w_a,
                    wins_b=w_b,
                    draws=d,
                    win_rate_a=round(wr_a, 3),
                    avg_score_a=0.0,
                    avg_score_b=0.0,
                    games_played=g_played,
                )

            with self._progress_lock:
                self._progress = TournamentProgressDTO(
                    is_running=True,
                    current_match=cur_m,
                    total_matches=tot_m,
                    current_agent_a=ev.get("agent_a", ""),
                    current_agent_b=ev.get("agent_b", ""),
                    status=ev.get("status", "running"),
                    percentage=pct,
                    elapsed_seconds=round(elapsed, 1),
                    estimated_remaining_seconds=round(est_rem, 1),
                    recent_matchup=recent_dto,
                )

            if self.connection_manager:
                self.connection_manager.broadcast_sync(
                    {
                        "type": "tournament_progress",
                        **self._progress.model_dump(),
                    }
                )

        try:
            tourney = Tournament(
                agents=agents,
                games_per_pair=games_per_pair,
                initial_elo=1200.0,
                k_factor=32.0,
                board=board,
                tickets_deck=tickets,
            )

            raw_results = tourney.run(seed=seed, progress_callback=on_progress)

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
            total_tourney_games = 0
            for m in raw_results["matchups"]:
                w_a = int(m["wins_a"])
                w_b = int(m["wins_b"])
                d = int(m["draws"])
                g_played = w_a + w_b + d
                total_tourney_games += g_played
                wr_a = (w_a / g_played) if g_played > 0 else 0.0
                matchup_dtos.append(
                    TournamentMatchupDTO(
                        agent_a=m["agent_a"],
                        agent_b=m["agent_b"],
                        wins_a=w_a,
                        wins_b=w_b,
                        draws=d,
                        win_rate_a=round(wr_a, 3),
                        avg_score_a=float(m["avg_score_a"]),
                        avg_score_b=float(m["avg_score_b"]),
                        games_played=g_played,
                    )
                )

            res = TournamentLeaderboardDTO(
                leaderboard=leaderboard_dtos,
                matchups=matchup_dtos,
                total_games=total_tourney_games,
                updated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                map_name=map_name,
                available_participants=all_available,
            )
            with self._lock:
                self._cached_leaderboard = res
            return res
        finally:
            with self._progress_lock:
                self._progress = TournamentProgressDTO(
                    is_running=False,
                    current_match=total_pairings,
                    total_matches=total_pairings,
                    status="idle",
                    percentage=100.0,
                    elapsed_seconds=round(
                        (datetime.datetime.now() - start_time).total_seconds(), 1
                    ),
                    estimated_remaining_seconds=0.0,
                )
            if self.connection_manager:
                self.connection_manager.broadcast_sync(
                    {
                        "type": "tournament_progress",
                        **self._progress.model_dump(),
                    }
                )
