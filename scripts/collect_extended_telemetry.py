#!/usr/bin/env python3
"""
Extended Gameplay Telemetry Collector for TicketToRide RL Lab.

Collects fine-grained causal and strategic metrics across simulated matches:
1. Initial Ticket Steiner Synergy Index vs Final Score / Win Probability.
2. Card Hoarding Threshold (Hand size at first claim) vs Route Block Frequency.
3. Mid-Game Ticket Draw Timing (Remaining trains, completion state) vs Completion Rate & Win Rate.
4. Route Contention Frequency (Chokepoints where both agents contest the same route).
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.game import Game
from src.game.graph import check_ticket_completed
from src.game.maps import load_usa_board
from src.game.player import Player
from src.game.route import Route
from src.game.state import GameState, TurnState
from src.game.ticket import DestinationTicket


def compute_shortest_path_routes(
    board: Board, city_start: str, city_target: str, claimed_by_opponent: set[str] | None = None
) -> list[Route]:
    """Dijkstra shortest path on board ignoring routes claimed by opponent."""
    if claimed_by_opponent is None:
        claimed_by_opponent = set()

    dist: dict[str, float] = {city_start: 0.0}
    prev_route: dict[str, Route | None] = {city_start: None}
    prev_city: dict[str, str | None] = {city_start: None}
    pq: list[tuple[float, str]] = [(0.0, city_start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == city_target:
            break

        for r in board.get_adjacent_routes(u):
            if r.id in claimed_by_opponent:
                continue

            v = r.city_b if r.city_a == u else r.city_a
            weight = float(r.length)

            if dist.get(u, float("inf")) + weight < dist.get(v, float("inf")):
                dist[v] = dist[u] + weight
                prev_route[v] = r
                prev_city[v] = u
                heapq.heappush(pq, (dist[v], v))

    if city_target not in dist or dist[city_target] == float("inf"):
        return []

    path: list[Route] = []
    curr = city_target
    while curr != city_start:
        r_obj = prev_route.get(curr)
        if not r_obj:
            break
        path.append(r_obj)
        next_city = prev_city.get(curr)
        if not next_city:
            break
        curr = next_city

    return path


def compute_steiner_synergy(board: Board, tickets: list[DestinationTicket]) -> float:
    """
    Computes the Steiner synergy ratio for a set of destination tickets:
    S = 1 - (Length(Union of shortest paths) / Sum(Individual shortest path lengths))
    Higher S in [0, 1) means high route sharing / corridor alignment.
    """
    if not tickets or len(tickets) <= 1:
        return 0.0

    individual_lengths = 0
    union_route_ids: set[str] = set()

    for t in tickets:
        path = compute_shortest_path_routes(board, t.city_a, t.city_b)
        path_len = sum(r.length for r in path)
        individual_lengths += path_len
        for r in path:
            union_route_ids.add(r.id)

    if individual_lengths == 0:
        return 0.0

    union_length = sum(board.get_route(rid).length for rid in union_route_ids if board.get_route(rid))
    synergy = 1.0 - (union_length / float(individual_lengths))
    return max(0.0, float(synergy))


class ExtendedTelemetryRunner:
    """Runs tournament matches while recording causal strategy events."""

    def __init__(self, board: Board, tickets: list[DestinationTicket]) -> None:
        self.board = board
        self.tickets = tickets

    def run_telemetry_matches(
        self,
        agent_pairs: list[tuple[BaseAgent, BaseAgent]],
        games_per_pair: int = 40,
        base_seed: int = 42,
    ) -> dict[str, Any]:
        synergy_records: list[dict[str, Any]] = []
        hoarding_records: list[dict[str, Any]] = []
        midgame_draw_records: list[dict[str, Any]] = []
        route_contention: dict[str, int] = defaultdict(int)
        route_claimed_counts: dict[str, int] = defaultdict(int)

        total_matches = 0
        start_time = time.perf_counter()

        for a1, a2 in agent_pairs:
            print(f"--> Simulating {games_per_pair * 2} games: {a1.name} vs {a2.name}...")
            for seed_offset in range(games_per_pair):
                # Run two symmetric games per seed offset (P0/P1 swapped)
                for p0, p1 in [(a1, a2), (a2, a1)]:
                    total_matches += 1
                    seed = base_seed + total_matches
                    game = Game(board=self.board, tickets_deck=self.tickets, num_players=2)
                    state = game.reset(seed=seed)

                    agents = [p0, p1]
                    first_claim_turn: dict[int, int | None] = {0: None, 1: None}
                    first_claim_hand: dict[int, int | None] = {0: None, 1: None}
                    blocked_routes_count: dict[int, int] = {0: 0, 1: 0}
                    initial_synergy: dict[int, float] = {}

                    # Pre-calculate active shortest paths for contention tracking
                    active_paths: dict[int, set[str]] = {0: set(), 1: set()}

                    turn = 0
                    while not state.is_game_over and state.turn_number < 250:
                        turn += 1
                        curr_p_idx = state.current_player_index
                        curr_player = state.current_player
                        if not curr_player:
                            break

                        valid_actions = game.valid_actions()
                        if not valid_actions:
                            break

                        # Record initial synergy once initial tickets are chosen
                        if state.turn_state not in [TurnState.CHOOSING_INITIAL_TICKETS] and curr_p_idx not in initial_synergy:
                            initial_synergy[curr_p_idx] = compute_steiner_synergy(self.board, curr_player.tickets)

                        # Update active target paths for curr_player
                        opp_claimed = set()
                        for p in state.players:
                            if p.id != curr_player.id:
                                opp_claimed.update(p.claimed_route_ids)

                        player_routes = [
                            r for rid in curr_player.claimed_route_ids if (r := self.board.get_route(rid)) is not None
                        ]
                        incomplete_tickets = [
                            t for t in curr_player.tickets if not check_ticket_completed(player_routes, t)
                        ]
                        curr_targets: set[str] = set()
                        for t in incomplete_tickets:
                            p_routes = compute_shortest_path_routes(self.board, t.city_a, t.city_b, opp_claimed)
                            for r in p_routes:
                                if r.id not in curr_player.claimed_route_ids:
                                    curr_targets.add(r.id)
                        active_paths[curr_p_idx] = curr_targets

                        # Detect Contention: routes desired by BOTH players
                        overlap = active_paths[0].intersection(active_paths[1])
                        for rid in overlap:
                            route_contention[rid] += 1

                        agent = agents[curr_p_idx]
                        action = agent.act(state, valid_actions, self.board)

                        # Metric 2: Card Hoarding & First Claim Turn
                        if action.action_type == ActionType.CLAIM_ROUTE:
                            if first_claim_turn[curr_p_idx] is None:
                                first_claim_turn[curr_p_idx] = turn
                                first_claim_hand[curr_p_idx] = sum(curr_player.cards.values())

                            # Check if this claim blocked the opponent's desired routes
                            opp_idx = 1 - curr_p_idx
                            if action.route_id in active_paths[opp_idx]:
                                blocked_routes_count[opp_idx] += 1

                            if action.route_id:
                                route_claimed_counts[action.route_id] += 1

                        # Metric 3: Mid-Game Ticket Draw Timing
                        if action.action_type == ActionType.DRAW_TICKETS:
                            completed_count = len(curr_player.tickets) - len(incomplete_tickets)
                            midgame_draw_records.append({
                                "agent": agent.name,
                                "turn": turn,
                                "trains_remaining": curr_player.trains_remaining,
                                "hand_size": sum(curr_player.cards.values()),
                                "completed_tickets_ratio": (
                                    completed_count / len(curr_player.tickets) if curr_player.tickets else 0.0
                                ),
                                "all_current_completed": len(incomplete_tickets) == 0,
                                "player_index": curr_p_idx,
                                "game_seed": seed,
                            })

                        state = game.step(action)

                    # Post-game analysis
                    scores = [state.players[0].score, state.players[1].score]
                    winner_idx = 0 if scores[0] > scores[1] else (1 if scores[1] > scores[0] else -1)

                    for p_idx in [0, 1]:
                        syn = initial_synergy.get(p_idx, 0.0)
                        is_win = 1 if winner_idx == p_idx else (0 if winner_idx != -1 else 0.5)
                        synergy_records.append({
                            "agent": agents[p_idx].name,
                            "synergy_index": syn,
                            "final_score": scores[p_idx],
                            "is_win": is_win,
                        })

                        hoarding_records.append({
                            "agent": agents[p_idx].name,
                            "first_claim_turn": first_claim_turn[p_idx] or turn,
                            "first_claim_hand": first_claim_hand[p_idx] or sum(state.players[p_idx].cards.values()),
                            "blocked_routes_experienced": blocked_routes_count[p_idx],
                            "is_win": is_win,
                        })

        elapsed = time.perf_counter() - start_time
        print(f"Finished {total_matches} matches in {elapsed:.2f}s ({total_matches / elapsed:.1f} games/s)")

        # Compile summaries
        # 1. Synergy Correlation
        high_syn_wins = [r["is_win"] for r in synergy_records if r["synergy_index"] >= 0.35]
        low_syn_wins = [r["is_win"] for r in synergy_records if r["synergy_index"] < 0.35]

        # 2. Hoarding correlation
        hoard_patient_blocks = [r["blocked_routes_experienced"] for r in hoarding_records if r["first_claim_hand"] >= 12]
        hoard_rush_blocks = [r["blocked_routes_experienced"] for r in hoarding_records if r["first_claim_hand"] < 8]

        # 3. Midgame ticket draw win rate by train count
        draws_abundant = [r for r in midgame_draw_records if r["trains_remaining"] >= 16]
        draws_scarce = [r for r in midgame_draw_records if r["trains_remaining"] < 16]

        # 4. Top Contended Routes
        top_contended = sorted(
            [
                {
                    "route_id": rid,
                    "cities": f"{self.board.get_route(rid).city_a}--{self.board.get_route(rid).city_b}",
                    "contention_count": count,
                    "claimed_count": route_claimed_counts.get(rid, 0),
                }
                for rid, count in route_contention.items()
                if self.board.get_route(rid)
            ],
            key=lambda x: x["contention_count"],
            reverse=True,
        )[:10]

        summary = {
            "total_matches": total_matches,
            "elapsed_seconds": elapsed,
            "synergy_stats": {
                "high_synergy_winrate": float(np.mean(high_syn_wins)) if high_syn_wins else 0.0,
                "low_synergy_winrate": float(np.mean(low_syn_wins)) if low_syn_wins else 0.0,
                "sample_high": len(high_syn_wins),
                "sample_low": len(low_syn_wins),
            },
            "hoarding_stats": {
                "patient_mean_blocks_suffered": float(np.mean(hoard_patient_blocks)) if hoard_patient_blocks else 0.0,
                "rush_mean_blocks_suffered": float(np.mean(hoard_rush_blocks)) if hoard_rush_blocks else 0.0,
                "sample_patient": len(hoard_patient_blocks),
                "sample_rush": len(hoard_rush_blocks),
            },
            "midgame_ticket_draws": {
                "total_midgame_draws": len(midgame_draw_records),
                "draws_abundant_trains_count": len(draws_abundant),
                "draws_scarce_trains_count": len(draws_scarce),
                "draws_when_all_completed_pct": float(
                    np.mean([1 if r["all_current_completed"] else 0 for r in midgame_draw_records])
                ) if midgame_draw_records else 0.0,
            },
            "top_contended_chokepoints": top_contended,
        }

        return {
            "summary": summary,
            "synergy_records": synergy_records[:100],  # sample for inspection
            "hoarding_records": hoarding_records[:100],
            "midgame_draw_records": midgame_draw_records[:100],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Extended Telemetry Collector")
    parser.add_argument("--games-per-pair", type=int, default=30, help="Games per pairing (default: 30)")
    parser.add_argument("--output", type=str, default="results/thesis/extended_telemetry_results.json")
    args = parser.parse_args()

    board, tickets = load_usa_board()

    # Agent lineup covering diverse strategic styles:
    strategic = StrategicAgent(name="Strategic_Dijkstra")
    greedy = GreedyAgent(name="Greedy_Score")
    random_ag = RandomAgent(name="Random_Baseline")
    mcts_agent = MCTSAgent(board=board, tickets=tickets, num_simulations=10, name="ISMCTS_10Sims")

    pairs = [
        (strategic, greedy),
        (strategic, mcts_agent),
        (strategic, random_ag),
        (greedy, mcts_agent),
    ]

    print("=" * 70)
    print("🚂 TICKETTORIDE RL LAB: EXTENDED GAMEPLAY TELEMETRY COLLECTOR")
    print("=" * 70)

    runner = ExtendedTelemetryRunner(board=board, tickets=tickets)
    results = runner.run_telemetry_matches(pairs, games_per_pair=args.games_per_pair)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("📊 EMPIRICAL TELEMETRY SUMMARY REPORT")
    print("=" * 70)
    s = results["summary"]
    print(f"Total Matches Analyzed: {s['total_matches']}")
    print(f"\n1. Steiner Synergy Impact (Drafting Phase):")
    print(f"   - High Synergy (>= 0.35) Win Rate: {s['synergy_stats']['high_synergy_winrate'] * 100:.1f}% (N={s['synergy_stats']['sample_high']})")
    print(f"   - Low Synergy (< 0.35) Win Rate:  {s['synergy_stats']['low_synergy_winrate'] * 100:.1f}% (N={s['synergy_stats']['sample_low']})")

    print(f"\n2. Card Hoarding vs Route Blocking:")
    print(f"   - Patient Hoarders (>=12 cards before 1st claim) Mean Blocks Suffered: {s['hoarding_stats']['patient_mean_blocks_suffered']:.2f}")
    print(f"   - Rush Claimers (<8 cards before 1st claim) Mean Blocks Suffered:      {s['hoarding_stats']['rush_mean_blocks_suffered']:.2f}")

    print(f"\n3. Mid-Game Ticket Draw Statistics:")
    print(f"   - Total Mid-Game Ticket Draws Observed: {s['midgame_ticket_draws']['total_midgame_draws']}")
    print(f"   - Draws Executed When All Existing Tickets Completed: {s['midgame_ticket_draws']['draws_when_all_completed_pct'] * 100:.1f}%")

    print(f"\n4. Top-5 Contested Chokepoint Routes:")
    for i, c in enumerate(s["top_contended_chokepoints"][:5], 1):
        print(f"   {i}. {c['cities']} (Route ID: {c['route_id']}) -> Contested in {c['contention_count']} turns (Claimed {c['claimed_count']} times)")

    print(f"\nArtifact saved to: {out_file.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
