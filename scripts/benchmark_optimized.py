"""Optimized Benchmark Script for TicketToRide RL Lab."""

import json
import time
import numpy as np
import torch

from src.game.maps import load_usa_board
from src.game.game import Game
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.strategic_agent import StrategicAgent
from src.environment.env import TicketToRideEnv
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

    # 1. Pure Game Core: 100 full games (Greedy vs Strategic)
    board, tickets = load_usa_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    p0 = StrategicAgent(name="Strategic")
    p1 = GreedyAgent(name="Greedy")
    
    num_games = 100
    start = time.perf_counter()
    total_steps = 0
    for i in range(num_games):
        state = game.reset(seed=1000 + i)
        while not state.is_game_over and state.turn_number < 300:
            actions = game.valid_actions()
            if not actions:
                break
            curr_agent = p0 if state.current_player_index == 0 else p1
            act = curr_agent.act(state, actions, board)
            state = game.step(act)
            total_steps += 1
    duration = time.perf_counter() - start
    games_per_sec = num_games / duration
    steps_per_sec = total_steps / duration
    results["game_core"] = {
        "num_games": num_games,
        "total_steps": total_steps,
        "duration_sec": duration,
        "games_per_sec": games_per_sec,
        "steps_per_sec": steps_per_sec,
        "ms_per_step": (duration / total_steps) * 1000,
    }
    print(f"1. Game Core (100 USA Games Strategic vs Greedy):")
    print(f"   Duration: {duration:.3f}s | Throughput: {games_per_sec:.2f} games/s | {steps_per_sec:.1f} steps/s | {results['game_core']['ms_per_step']:.3f} ms/step")

    # 2. Observation Encoding
    encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)
    game.reset(seed=42)
    for _ in range(30):
        acts = game.valid_actions()
        if not acts:
            break
        game.step(p0.act(game.state, acts, board))
    
    n_encodes = 5000
    start = time.perf_counter()
    for _ in range(n_encodes):
        obs = encoder.encode(game.state, player_index=0)
    duration = time.perf_counter() - start
    encodes_per_sec = n_encodes / duration
    results["observation_encoder"] = {
        "n_encodes": n_encodes,
        "duration_sec": duration,
        "encodes_per_sec": encodes_per_sec,
        "us_per_encode": (duration / n_encodes) * 1e6,
    }
    print(f"2. ObservationV1 Encode ({n_encodes} calls):")
    print(f"   Duration: {duration:.3f}s | Throughput: {encodes_per_sec:.1f} calls/s | {results['observation_encoder']['us_per_encode']:.2f} µs/call")

    # 3. Action Masking
    discrete_actions = DiscreteActionSpace(board=board)
    masker = ActionMasker(discrete_actions)
    valid_acts = game.valid_actions()
    pending = game.state.players[0].pending_tickets

    n_masks = 10000
    start = time.perf_counter()
    for _ in range(n_masks):
        mask = masker.compute_mask(valid_acts, pending_tickets=pending)
    duration = time.perf_counter() - start
    masks_per_sec = n_masks / duration
    results["action_masker"] = {
        "n_masks": n_masks,
        "duration_sec": duration,
        "masks_per_sec": masks_per_sec,
        "us_per_mask": (duration / n_masks) * 1e6,
    }
    print(f"3. ActionMasker ({n_masks} calls):")
    print(f"   Duration: {duration:.3f}s | Throughput: {masks_per_sec:.1f} calls/s | {results['action_masker']['us_per_mask']:.2f} µs/call")

    # 4. Reward Calculator
    reward_calc = DefaultRewardCalculator(board=board)
    prev_st = game.state
    acts = game.valid_actions()
    act = acts[0] if acts else None
    next_st = game.state

    n_rewards = 10000
    start = time.perf_counter()
    for _ in range(n_rewards):
        r = reward_calc.calculate(prev_st, act, next_st, player_index=0)
    duration = time.perf_counter() - start
    rewards_per_sec = n_rewards / duration
    results["reward_calculator"] = {
        "n_rewards": n_rewards,
        "duration_sec": duration,
        "rewards_per_sec": rewards_per_sec,
        "us_per_calc": (duration / n_rewards) * 1e6,
    }
    print(f"4. RewardCalculator ({n_rewards} calls):")
    print(f"   Duration: {duration:.3f}s | Throughput: {rewards_per_sec:.1f} calls/s | {results['reward_calculator']['us_per_calc']:.2f} µs/call")

    # 5. PPO Training (2048 steps rollout & train epochs)
    env_ppo = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=GreedyAgent(name="GreedyBot"), num_players=2)
    ppo_config = {
        "lr": 3e-4,
        "rollout_steps": 256,
        "minibatch_size": 32,
        "num_epochs": 4,
    }
    trainer_ppo = MaskedPPOTrainer(env=env_ppo, config=ppo_config)

    n_ppo_steps = 2048
    start = time.perf_counter()
    steps_done = 0
    while steps_done < n_ppo_steps:
        r_info = trainer_ppo.collect_rollout()
        metrics = trainer_ppo.train_epoch()
        steps_done += trainer_ppo.rollout_steps
    duration = time.perf_counter() - start
    ppo_fps = steps_done / duration
    results["ppo_training"] = {
        "steps": steps_done,
        "duration_sec": duration,
        "fps": ppo_fps,
        "ms_per_step": (duration / steps_done) * 1000,
    }
    print(f"5. Masked PPO Training ({steps_done} steps vs Greedy):")
    print(f"   Duration: {duration:.3f}s | FPS: {ppo_fps:.1f} steps/s | {results['ppo_training']['ms_per_step']:.3f} ms/step")

    # 6. DQN Training (2000 steps step & train_step)
    env_dqn = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(name="RandomBot"), num_players=2)
    dqn_config = {
        "lr": 5e-4,
        "batch_size": 32,
        "buffer_size": 10000,
        "learning_starts": 100,
    }
    trainer_dqn = MaskedDQNTrainer(env=env_dqn, config=dqn_config)

    n_dqn_steps = 2000
    start = time.perf_counter()
    for s in range(n_dqn_steps):
        trainer_dqn.step()
        trainer_dqn.train_step()
    duration = time.perf_counter() - start
    dqn_fps = n_dqn_steps / duration
    results["dqn_training"] = {
        "steps": n_dqn_steps,
        "duration_sec": duration,
        "fps": dqn_fps,
        "ms_per_step": (duration / n_dqn_steps) * 1000,
    }
    print(f"6. Masked DQN Training ({n_dqn_steps} steps vs Random):")
    print(f"   Duration: {duration:.3f}s | FPS: {dqn_fps:.1f} steps/s | {results['dqn_training']['ms_per_step']:.3f} ms/step")

    # 7. Tournament Benchmark (300 games)
    tournament_agents = [
        RandomAgent(seed=42, name="Rand"),
        GreedyAgent(name="Greed"),
        StrategicAgent(name="Strat"),
    ]
    tourney = Tournament(agents=tournament_agents, games_per_pair=100)
    start = time.perf_counter()
    tourney_res = tourney.run(seed=123)
    duration = time.perf_counter() - start
    total_tourney_games = 300
    tourney_gps = total_tourney_games / duration
    results["tournament"] = {
        "games": total_tourney_games,
        "duration_sec": duration,
        "games_per_sec": tourney_gps,
    }
    print(f"7. Tournament (300 games Round Robin):")
    print(f"   Duration: {duration:.3f}s | Throughput: {tourney_gps:.2f} games/s")
    print("=" * 60)

    with open("benchmark_optimized.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to benchmark_optimized.json")


if __name__ == "__main__":
    run_all_benchmarks()
