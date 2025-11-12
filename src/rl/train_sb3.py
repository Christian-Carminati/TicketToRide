"""Training entry-point for Stable Baselines3 agents on Ticket to Ride."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("stable-baselines3 is required to run this training script") from exc

from map import TicketToRideMap

from rl.envs import RewardConfig, SingleAgentTicketToRideEnv
from rl.opponents import build_opponent

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_DIR = DEFAULT_PROJECT_ROOT / "src" / "map"
DEFAULT_TICKETS = DEFAULT_MAP_DIR / "tickets.csv"
DEFAULT_ROUTES = DEFAULT_MAP_DIR / "routes.csv"
DEFAULT_LOCATIONS = DEFAULT_MAP_DIR / "city_locations.json"


def load_graph(city_locations: Path, routes_file: Path):
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
) -> SingleAgentTicketToRideEnv:
    graph = load_graph(city_locations, routes_file)
    opponent = build_opponent(opponent_name, graph=graph, tickets_file=str(tickets_file))
    return SingleAgentTicketToRideEnv(
        graph,
        str(tickets_file),
        opponent_policy=opponent,
        reward_config=reward_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable Baselines3 agents on Ticket to Ride")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo", help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--opponent", type=str, default="greedy", help="Opponent policy name")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/sb3"), help="Directory for logs and checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=25, help="Episodes used during periodic evaluation")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Frequency (steps) for saving checkpoints")
    parser.add_argument("--reward-efficiency", type=float, default=0.5, help="Weight for efficiency component")
    parser.add_argument("--reward-ticket", type=float, default=0.05, help="Weight for ticket component")
    parser.add_argument("--reward-connectivity", type=float, default=5.0, help="Weight for connectivity bonus")
    parser.add_argument(
        "--reward-final-scale",
        type=float,
        default=0.1,
        help="Multiplier applied to the final score difference at episode end",
    )
    parser.add_argument(
        "--city-locations",
        type=Path,
        default=DEFAULT_LOCATIONS,
        help="Path to the JSON file with city coordinates",
    )
    parser.add_argument(
        "--routes-file",
        type=Path,
        default=DEFAULT_ROUTES,
        help="Path to the CSV file containing route definitions",
    )
    parser.add_argument(
        "--tickets-file",
        type=Path,
        default=DEFAULT_TICKETS,
        help="Path to the CSV file containing destination tickets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device passed to Stable Baselines3 (cpu, cuda, etc.)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional custom learning rate for the selected algorithm",
    )
    return parser.parse_args()


def setup_callbacks(
    env_factory: Callable[[], SingleAgentTicketToRideEnv],
    log_dir: Path,
    eval_episodes: int,
    checkpoint_freq: int,
) -> tuple[EvalCallback, CheckpointCallback]:
    eval_env = Monitor(env_factory())
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=checkpoint_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=eval_episodes,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ttr_agent",
    )
    return eval_callback, checkpoint_callback


def build_model(
    algorithm: str,
    env,
    *,
    learning_rate: Optional[float],
    device: str,
    seed: int,
):
    if algorithm == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate or 3e-4,
            device=device,
            seed=seed,
        )
    if algorithm == "dqn":
        return DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate or 1e-3,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            target_update_interval=1_000,
            device=device,
            seed=seed,
        )
    raise ValueError(f"Unsupported algorithm '{algorithm}'")


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    reward_config = RewardConfig(
        efficiency_weight=args.reward_efficiency,
        ticket_weight=args.reward_ticket,
        connectivity_bonus=args.reward_connectivity,
        final_score_scale=args.reward_final_scale,
    )

    def env_factory() -> SingleAgentTicketToRideEnv:
        return make_env(
            city_locations=args.city_locations,
            routes_file=args.routes_file,
            tickets_file=args.tickets_file,
            opponent_name=args.opponent,
            reward_config=reward_config,
        )

    env = Monitor(env_factory())
    eval_callback, checkpoint_callback = setup_callbacks(
        env_factory,
        log_dir,
        args.eval_episodes,
        max(1, args.checkpoint_freq),
    )

    model = build_model(
        args.algorithm,
        env,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback])

    model_path = log_dir / f"{args.algorithm}_ttr_final"
    model.save(str(model_path))
    print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
