"""High-performance deterministic evaluation framework."""

from src.agents.base_agent import BaseAgent
from src.evaluation.metrics import EvaluationMetrics
from src.game.board import Board
from src.game.game import Game
from src.game.graph import check_ticket_completed
from src.game.ticket import DestinationTicket


class Evaluator:
    """Evaluates two agents in head-to-head matches with alternating player positions."""

    def __init__(
        self,
        board: Board | None = None,
        tickets_deck: list[DestinationTicket] | None = None,
        max_turns: int = 400,
        seed: int = 42,
    ) -> None:
        self.board = board
        self.tickets_deck = tickets_deck
        self.max_turns = max_turns
        self.seed = seed

    def evaluate(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        num_games: int = 100,
        seed: int | None = None,
    ) -> dict[str, EvaluationMetrics]:
        """Run N deterministic head-to-head games between agent_a and agent_b."""
        run_seed = self.seed if seed is None else seed
        metrics_a = EvaluationMetrics(total_games=num_games)
        metrics_b = EvaluationMetrics(total_games=num_games)

        total_score_a = 0
        total_score_b = 0
        total_turns = 0
        tickets_drawn_a = 0
        tickets_completed_a = 0
        tickets_drawn_b = 0
        tickets_completed_b = 0

        for game_idx in range(num_games):
            game_seed = run_seed + game_idx
            # Alternate player seats: even games -> (A is player 0, B is player 1)
            #                       odd games -> (B is player 0, A is player 1)
            is_a_first = (game_idx % 2 == 0)
            p0_agent = agent_a if is_a_first else agent_b
            p1_agent = agent_b if is_a_first else agent_a

            agent_a.reset(seed=game_seed)
            agent_b.reset(seed=game_seed + 100000)

            game = Game(
                board=self.board,
                tickets_deck=self.tickets_deck,
                num_players=2,
                seed=game_seed,
            )
            game.reset(seed=game_seed)

            # Play until game over or max turns safety bound
            while not game.state.is_game_over and game.state.turn_number < self.max_turns:
                curr_idx = game.state.current_player_index
                curr_agent = p0_agent if curr_idx == 0 else p1_agent

                valid_actions = game.valid_actions()
                if not valid_actions:
                    break

                action = curr_agent.act(game.state, valid_actions, game.board)
                game.step(action)

            # Extract end game statistics
            p0 = game.state.players[0]
            p1 = game.state.players[1]
            score_p0 = p0.score
            score_p1 = p1.score

            score_a = score_p0 if is_a_first else score_p1
            score_b = score_p1 if is_a_first else score_p0

            total_score_a += score_a
            total_score_b += score_b
            total_turns += game.state.turn_number

            # Ticket stats
            p_a = p0 if is_a_first else p1
            p_b = p1 if is_a_first else p0
            routes_a = [
                game.board.get_route(rid)
                for rid in p_a.claimed_route_ids
                if game.board.get_route(rid) is not None
            ]
            routes_b = [
                game.board.get_route(rid)
                for rid in p_b.claimed_route_ids
                if game.board.get_route(rid) is not None
            ]

            tickets_drawn_a += len(p_a.tickets)
            tickets_drawn_b += len(p_b.tickets)
            tickets_completed_a += sum(
                1 for t in p_a.tickets if check_ticket_completed(routes_a, t)
            )
            tickets_completed_b += sum(
                1 for t in p_b.tickets if check_ticket_completed(routes_b, t)
            )

            # Record game outcome
            if score_a > score_b:
                metrics_a.wins += 1
                metrics_b.losses += 1
            elif score_b > score_a:
                metrics_b.wins += 1
                metrics_a.losses += 1
            else:
                metrics_a.draws += 1
                metrics_b.draws += 1

        # Aggregate metrics
        if num_games > 0:
            metrics_a.avg_score = total_score_a / num_games
            metrics_b.avg_score = total_score_b / num_games
            metrics_a.avg_score_diff = (total_score_a - total_score_b) / num_games
            metrics_b.avg_score_diff = (total_score_b - total_score_a) / num_games
            metrics_a.avg_turns = total_turns / num_games
            metrics_b.avg_turns = total_turns / num_games
            metrics_a.ticket_completion_rate = (
                (tickets_completed_a / tickets_drawn_a) if tickets_drawn_a > 0 else 0.0
            )
            metrics_b.ticket_completion_rate = (
                (tickets_completed_b / tickets_drawn_b) if tickets_drawn_b > 0 else 0.0
            )

        return {agent_a.name: metrics_a, agent_b.name: metrics_b}

    def evaluate_head_to_head(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        num_games: int = 100,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Convenience method returning summary win rate and score dictionary."""
        results = self.evaluate(agent_a, agent_b, num_games=num_games, seed=seed)
        metrics_a = results[agent_a.name]
        metrics_b = results[agent_b.name]

        return {
            "agent_a_win_rate": float(metrics_a.win_rate),
            "agent_b_win_rate": float(metrics_b.win_rate),
            "agent_a_mean_score": float(metrics_a.avg_score),
            "agent_b_mean_score": float(metrics_b.avg_score),
            "draw_rate": float(metrics_a.draws / max(1, num_games)),
        }
