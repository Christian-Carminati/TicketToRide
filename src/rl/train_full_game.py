"""
Training script for full game RL environment with card mechanics.

This script trains agents using the FullGameSingleAgentEnv which includes
complete card management, deck, face-up cards, and proper route validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise ImportError("stable-baselines3 is required to run this training script") from exc

from map import TicketToRideMap
from rl.full_game_env import FullGameSingleAgentEnv,  RewardConfig
from rl.opponents import build_opponent
from gymnasium.wrappers import TimeLimit
try:
    # Supersuit is optional; only needed for PettingZoo self-play with SB3
    import supersuit as ss  # type: ignore[import-not-found]
    from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1  # type: ignore[import-not-found]
    SUPERSUIT_AVAILABLE = True
except Exception:
    SUPERSUIT_AVAILABLE = False
try:
    from stable_baselines3.common.vec_env import VecMonitor  # type: ignore[attr-defined]
except Exception:
    VecMonitor = None  # type: ignore[assignment]


def load_graph(city_locations: Path, routes_file: Path):
    """Load the game graph."""
    ttr_map = TicketToRideMap()
    ttr_map.load_graph(str(city_locations), str(routes_file))
    return ttr_map.get_graph()


def make_env(
    *,
    city_locations: Path,
    routes_file: Path,
    tickets_file: Path,
    opponent_name: str,
    reward_config: RewardConfig,
    max_episode_steps: int,
    env_type: str = "single",
) -> FullGameSingleAgentEnv:
    """Create a full game environment."""
    graph = load_graph(city_locations, routes_file)
    if env_type == "single":
        opponent = build_opponent(opponent_name, graph=graph, tickets_file=str(tickets_file))
        base_env = FullGameSingleAgentEnv(
            graph, str(tickets_file), opponent_policy=opponent, reward_config=reward_config
        )
        env = TimeLimit(base_env, max_episode_steps=max_episode_steps)
        return Monitor(env)
    elif env_type == "selfplay":
        # Build PettingZoo AEC env for self-play and convert to SB3 VecEnv via Supersuit
        if not SUPERSUIT_AVAILABLE or VecMonitor is None:
            raise RuntimeError(
                "Self-play selected but Supersuit/VecMonitor not available.\n"
                "Please install dependencies:\n"
                "  pip install pettingzoo supersuit stable-baselines3"
            )
        from rl.full_game_env import FullGameSelfPlayEnv
        from pettingzoo.utils.conversions import aec_to_parallel
        
        # Create the base AEC environment with max_cycles parameter if supported
        # PettingZoo AEC environments typically accept max_cycles in their constructor
        try:
            aec = FullGameSelfPlayEnv(
                graph, 
                str(tickets_file), 
                reward_config=reward_config,
                max_cycles=max_episode_steps
            )
        except TypeError:
            # If max_cycles isn't supported, create without it
            # The environment should have its own termination logic
            aec = FullGameSelfPlayEnv(graph, str(tickets_file), reward_config=reward_config)
            print(f"Note: FullGameSelfPlayEnv doesn't accept max_cycles parameter. "
                  f"Using environment's built-in termination conditions.")
        
        # Convert AEC to ParallelEnv
        parallel_env = aec_to_parallel(aec)
        
        # Convert to SB3-compatible VecEnv
        vec = pettingzoo_env_to_vec_env_v1(parallel_env)
        vec = concat_vec_envs_v1(vec, num_vec_envs=1,  base_class="stable_baselines3")
        
        # Add VecMonitor for logging
        return VecMonitor(vec)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent on full Ticket to Ride game")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="greedy",
        choices=["random", "greedy", "heuristic"],
        help="Opponent policy",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/full_game"),
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50000,
        help="Frequency of checkpoint saves",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("map"),
        help="Directory containing map files",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="single",
        choices=["single", "selfplay"],
        help="Environment type: single-agent vs PettingZoo self-play",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Safety cap to truncate very long episodes (prevents training stalls)",
    )

    args = parser.parse_args()

    # Setup paths
    city_locations = args.map_dir / "city_locations.json"
    routes_file = args.map_dir / "routes.csv"
    tickets_file = args.map_dir / "tickets.csv"

    # Reward configuration
    reward_config = RewardConfig(
        invalid_action_penalty=10.0,
        efficiency_weight=0.5,
        connectivity_bonus=5.0,
        ticket_weight=0.05,
        final_score_scale=0.1,
        card_draw_bonus=0.1,
    )

    # Create environment
    print("Creating environment...")
    env = make_env(
        city_locations=city_locations,
        routes_file=routes_file,
        tickets_file=tickets_file,
        opponent_name=args.opponent,
        reward_config=reward_config,
        max_episode_steps=args.max_episode_steps,
        env_type=args.env_type,
    )

    # Create log directory
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print(f"Creating {args.algorithm.upper()} model...")
    if args.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=2,
            tensorboard_log=str(args.log_dir / "tensorboard"),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    else:  # DQN
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(args.log_dir / "tensorboard"),
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.log_dir / "checkpoints"),
        name_prefix="model",
    )

    # Evaluation environment
    eval_env = make_env(
        city_locations=city_locations,
        routes_file=routes_file,
        tickets_file=tickets_file,
        opponent_name=args.opponent,
        reward_config=reward_config,
        max_episode_steps=args.max_episode_steps,
        env_type=args.env_type,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval_logs"),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train
    print(f"Training for {args.timesteps} timesteps...")
    print(f"Opponent: {args.opponent}")
    print(f"Log directory: {args.log_dir}")
    print(f"Env type: {args.env_type}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    # num_routes/num_actions available on underlying base env in single mode
    try:
        print(f"Number of routes: {env.unwrapped.num_routes}")  # type: ignore[attr-defined]
        print(f"Number of actions: {env.unwrapped.num_actions}")  # type: ignore[attr-defined]
    except Exception:
        pass
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_model_path = args.log_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nTraining complete! Model saved to {final_model_path}")

    # Test the trained model
    print("\nTesting trained model...")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 1000

    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"Test episode: {steps} steps, total reward: {total_reward:.2f}")
    if "agent_score" in info:
        print(f"Agent score: {info['agent_score']}")
        print(f"Opponent score: {info['opponent_score']}")


if __name__ == "__main__":
    main()

