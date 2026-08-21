"""Behavioral profiling and strategic metrics evaluation for RL agents."""

from dataclasses import asdict, dataclass
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.game.action import ActionType
from src.game.board import Board
from src.game.game import Game
from src.game.graph import check_tickets_completed_batch
from src.game.rules import GameRules
from src.game.ticket import DestinationTicket


@dataclass
class BehavioralProfile:
    """Comprehensive behavioral signature of an agent across an evaluation match set."""

    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0
    avg_score: float = 0.0
    avg_score_diff: float = 0.0
    avg_routes_claimed: float = 0.0
    avg_route_length: float = 0.0
    route_efficiency: float = 0.0
    ticket_completion_rate: float = 0.0
    tickets_completed_avg: float = 0.0
    tickets_drawn_avg: float = 0.0
    ticket_penalty_avg: float = 0.0
    avg_game_turns: float = 0.0
    avg_game_length_turns: float = 0.0
    cards_drawn_ratio: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


class BehavioralEvaluator:
    """Simulates matches and extracts fine-grained strategic and behavioral metrics."""

    def __init__(
        self,
        board: Board | None = None,
        tickets_deck: list[DestinationTicket] | None = None,
        max_turns: int = 400,
        seed: int = 42,
        tickets: list[DestinationTicket] | None = None,
    ) -> None:
        self.board = board
        self.tickets_deck = tickets_deck if tickets_deck is not None else tickets
        self.max_turns = max_turns
        self.seed = seed

    def _resolve_opponent(self, opponent: BaseAgent | str, seed: int) -> BaseAgent:
        if isinstance(opponent, BaseAgent):
            return opponent

        opp_str = str(opponent).lower()
        if opp_str == "random":
            return RandomAgent(seed=seed, name="RandomOpponent")
        if opp_str == "greedy":
            return GreedyAgent(name="GreedyOpponent")
        if opp_str == "strategic":
            return StrategicAgent(name="StrategicOpponent")
        if opp_str == "heuristic":
            return HeuristicAgent(name="HeuristicOpponent")
        raise ValueError(f"Unknown opponent type: '{opponent}'")

    def profile_agent(
        self,
        agent: BaseAgent,
        opponent: BaseAgent | str,
        num_games: int = 20,
        seed: int | None = None,
    ) -> BehavioralProfile:
        """Run N games between agent and opponent, collecting fine-grained strategic metrics."""
        run_seed = self.seed if seed is None else seed
        opp_agent = self._resolve_opponent(opponent, seed=run_seed + 9999)

        profile = BehavioralProfile(total_games=num_games)

        total_score_agent = 0.0
        total_score_opp = 0.0
        total_turns = 0
        total_agent_moves = 0
        total_card_draw_moves = 0

        total_routes_claimed = 0
        total_route_length = 0
        total_route_points = 0
        total_trains_consumed = 0

        total_tickets_drawn = 0
        total_tickets_completed = 0
        total_ticket_penalties = 0.0

        for game_idx in range(num_games):
            game_seed = run_seed + game_idx
            is_agent_p0 = (game_idx % 2 == 0)
            p0_agent = agent if is_agent_p0 else opp_agent
            p1_agent = opp_agent if is_agent_p0 else agent

            agent.reset(seed=game_seed)
            opp_agent.reset(seed=game_seed + 10000)

            game = Game(
                board=self.board,
                tickets_deck=self.tickets_deck,
                num_players=2,
                seed=game_seed,
            )
            game.reset(seed=game_seed)

            # Step through game
            while not game.state.is_game_over and game.state.turn_number < self.max_turns:
                curr_idx = game.state.current_player_index
                curr_agent = p0_agent if curr_idx == 0 else p1_agent
                valid_actions = game.valid_actions()
                if not valid_actions:
                    break

                action = curr_agent.act(game.state, valid_actions, game.board)

                # Track action types for agent
                if (is_agent_p0 and curr_idx == 0) or (not is_agent_p0 and curr_idx == 1):
                    total_agent_moves += 1
                    if action.action_type in (ActionType.DRAW_VISIBLE_CARD, ActionType.DRAW_HIDDEN_CARD):
                        total_card_draw_moves += 1

                game.step(action)

            # Extract end-of-game data
            p_agent_idx = 0 if is_agent_p0 else 1
            p_opp_idx = 1 if is_agent_p0 else 0

            p_agent = game.state.players[p_agent_idx]
            p_opp = game.state.players[p_opp_idx]

            total_score_agent += float(p_agent.score)
            total_score_opp += float(p_opp.score)
            total_turns += game.state.turn_number

            # Route statistics for agent
            claimed_routes = [
                game.board.get_route(rid)
                for rid in p_agent.claimed_route_ids
                if game.board.get_route(rid) is not None
            ]
            total_routes_claimed += len(claimed_routes)
            for r in claimed_routes:
                total_route_length += r.length
                total_route_points += GameRules.points_for_route_length(r.length)
                total_trains_consumed += r.length

            # Ticket statistics for agent
            completed_dict = check_tickets_completed_batch(claimed_routes, p_agent.tickets)
            total_tickets_drawn += len(p_agent.tickets)
            completed_count = sum(1 for t in p_agent.tickets if completed_dict.get(t.id, False))
            total_tickets_completed += completed_count

            penalties = sum(float(t.points) for t in p_agent.tickets if not completed_dict.get(t.id, False))
            total_ticket_penalties += penalties

            # Win/Loss outcome
            if p_agent.score > p_opp.score:
                profile.wins += 1
            elif p_opp.score > p_agent.score:
                profile.losses += 1
            else:
                profile.draws += 1

        # Aggregate profile metrics
        if num_games > 0:
            profile.win_rate = float(profile.wins / num_games)
            profile.avg_score = float(total_score_agent / num_games)
            profile.avg_score_diff = float((total_score_agent - total_score_opp) / num_games)
            profile.avg_routes_claimed = float(total_routes_claimed / num_games)
            profile.avg_route_length = float(
                total_route_length / total_routes_claimed if total_routes_claimed > 0 else 0.0
            )
            profile.route_efficiency = float(
                total_route_points / total_trains_consumed if total_trains_consumed > 0 else 0.0
            )
            profile.tickets_drawn_avg = float(total_tickets_drawn / num_games)
            profile.tickets_completed_avg = float(total_tickets_completed / num_games)
            profile.ticket_completion_rate = float(
                total_tickets_completed / total_tickets_drawn if total_tickets_drawn > 0 else 0.0
            )
            profile.ticket_penalty_avg = float(total_ticket_penalties / num_games)
            profile.avg_game_turns = float(total_turns / num_games)
            profile.avg_game_length_turns = float(total_turns / num_games)
            profile.cards_drawn_ratio = float(
                total_card_draw_moves / total_agent_moves if total_agent_moves > 0 else 0.0
            )

        return profile

    def profile_multi_opponent(
        self,
        agent: BaseAgent,
        opponents: list[str | BaseAgent] | None = None,
        games_per_opponent: int = 20,
        seed: int | None = None,
    ) -> dict[str, BehavioralProfile]:
        """Profile agent against multiple baseline opponents."""
        opp_list = opponents or ["random", "greedy", "strategic"]
        results: dict[str, BehavioralProfile] = {}
        for opp in opp_list:
            opp_name = opp if isinstance(opp, str) else opp.name
            results[opp_name] = self.profile_agent(
                agent=agent,
                opponent=opp,
                num_games=games_per_opponent,
                seed=seed,
            )
        return results
