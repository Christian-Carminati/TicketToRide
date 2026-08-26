"""Optimized Benchmark Script for TicketToRide RL Lab measuring Native Rust Core & Parallel Engine."""

import json
import time
import numpy as np
import torch

from src.game.maps import load_usa_board
from src.game.game import Game
from src.game.native import NativeGame, NativeVectorEnv
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.strategic_agent import StrategicAgent
from src.environment.observation import ObservationV1
from src.environment.action_space import DiscreteActionSpace
from src.environment.action_mask import ActionMasker
from src.environment.reward import DefaultRewardCalculator
from src.rl.ppo import MaskedPPOTrainer
from src.rl.dqn import MaskedDQNTrainer
from src.evaluation.tournament import Tournament


def run_all_benchmarks():
    results = {}
    print("=" * 60)
    print("STARTING OPTIMIZED SCIENTIFIC BENCHMARK")
    print("=" * 60)

    # 1. Multi-threaded Native Vector Batch Simulation (Rayon GIL-free)
    vec_env = NativeVectorEnv(num_envs=64, base_seed=42)
    start = time.perf_counter()
    vec_steps = vec_env.step_batch_sim(num_steps_per_env=5000)
    vec_duration = time.perf_counter() - start
    vec_steps_per_sec = vec_steps / vec_duration
    # Approximately 70 steps per game
    estimated_games_per_sec = vec_steps_per_sec / 70.0
    results["native_vector_batch"] = {
        "num_envs": 64,
        "total_steps": vec_steps,
        "duration_sec": vec_duration,
        "steps_per_sec": vec_steps_per_sec,
        "games_per_sec": estimated_games_per_sec,
    }
    print(f"1. Multi-threaded Native Vector Env (64 parallel workers):")
    print(f"   Duration: {vec_duration:.3f}s | Throughput: {vec_steps_per_sec:,.0f} steps/s | ~{estimated_games_per_sec:,.0f} games/s")

    # 2. Native State Cloning (Stack memcpy)
    native_game = NativeGame(seed=42)
    n_clones = 500000
    start = time.perf_counter()
    for _ in range(n_clones):
        _ = native_game.clone()
    duration = time.perf_counter() - start
    clones_per_sec = n_clones / duration
    results["state_cloning"] = {
        "n_clones": n_clones,
        "duration_sec": duration,
        "clones_per_sec": clones_per_sec,
        "ns_per_clone": (duration / n_clones) * 1e9,
    }
    print(f"2. Native State Cloning ({n_clones} clones):")
    print(f"   Duration: {duration:.3f}s | Throughput: {clones_per_sec:,.1f} clones/s | {results['state_cloning']['ns_per_clone']:.2f} ns/clone")

    # 3. Observation Encoding
    board, tickets = load_usa_board()
    encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)
    p0 = StrategicAgent(name="Strategic")
    for _ in range(20):
        acts = game.valid_actions()
        if not acts:
            break
        game.step(p0.act(game.state, acts, board))

    n_encodes = 10000
    start = time.perf_counter()
    for _ in range(n_encodes):
        _ = encoder.encode(game.state, player_index=0)
    duration = time.perf_counter() - start
    encodes_per_sec = n_encodes / duration
    results["observation_encoder"] = {
        "n_encodes": n_encodes,
        "duration_sec": duration,
        "encodes_per_sec": encodes_per_sec,
        "us_per_encode": (duration / n_encodes) * 1e6,
    }
    print(f"3. ObservationV1 Encode ({n_encodes} calls):")
    print(f"   Duration: {duration:.3f}s | Throughput: {encodes_per_sec:.1f} calls/s | {results['observation_encoder']['us_per_encode']:.2f} µs/call")

    # 4. Action Masking
    discrete_actions = DiscreteActionSpace(board=board)
    masker = ActionMasker(discrete_actions)
    valid_acts = game.valid_actions()
    pending = game.state.players[0].pending_tickets

    n_masks = 20000
    start = time.perf_counter()
    for _ in range(n_masks):
        _ = masker.compute_mask(valid_acts, pending_tickets=pending)
    duration = time.perf_counter() - start
    masks_per_sec = n_masks / duration
    results["action_masker"] = {
        "n_masks": n_masks,
        "duration_sec": duration,
        "masks_per_sec": masks_per_sec,
        "us_per_mask": (duration / n_masks) * 1e6,
    }
    print(f"4. ActionMasker ({n_masks} calls):")
    print(f"   Duration: {duration:.3f}s | Throughput: {masks_per_sec:.1f} calls/s | {results['action_masker']['us_per_mask']:.2f} µs/call")

    # 5. Reward Calculation
    reward_calc = DefaultRewardCalculator(board=board)
    prev_st = game.state
    acts = game.valid_actions()
    act = acts[0] if acts else None
    next_st = game.state

    n_rewards = 20000
    start = time.perf_counter()
    for _ in range(n_rewards):
        _ = reward_calc.calculate(prev_st, act, next_st, player_index=0)
    duration = time.perf_counter() - start
    rewards_per_sec = n_rewards / duration
    results["reward_calculator"] = {
        "n_rewards": n_rewards,
        "duration_sec": duration,
        "rewards_per_sec": rewards_per_sec,
        "us_per_calc": (duration / n_rewards) * 1e6,
    }
    print(f"5. RewardCalculator ({n_rewards} calls):")
    print(f"   Duration: {duration:.3f}s | Throughput: {rewards_per_sec:.1f} calls/s | {results['reward_calculator']['us_per_calc']:.2f} µs/call")

    # 6. Parallel Tournament Execution (300 games)
    tournament_agents = [
        RandomAgent(seed=42, name="Rand"),
        GreedyAgent(name="Greed"),
        StrategicAgent(name="Strat"),
    ]
    tournament = Tournament(agents=tournament_agents, games_per_pair=100, board=board, tickets_deck=tickets)
    start = time.perf_counter()
    tourn_res = tournament.run_parallel(seed=123)
    duration = time.perf_counter() - start
    total_tourn_games = 300
    tourney_gps = total_tourn_games / duration
    results["tournament"] = {
        "games": total_tourn_games,
        "duration_sec": duration,
        "games_per_sec": tourney_gps,
    }
    print(f"6. Parallel Multi-Core Tournament ({total_tourn_games} games):")
    print(f"   Duration: {duration:.3f}s | Throughput: {tourney_gps:.2f} games/s")

    # Save to benchmark_optimized.json
    with open("benchmark_optimized.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("OPTIMIZED BENCHMARK COMPLETE -> saved to benchmark_optimized.json")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_all_benchmarks()
