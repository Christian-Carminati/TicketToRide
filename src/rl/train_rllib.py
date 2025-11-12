"""Training entry-point for RLlib multi-agent self-play on Ticket to Ride."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.utils.policy import PolicySpec
    from ray.tune.registry import register_env
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("ray[rllib] is required to run this training script") from exc

from map import TicketToRideMap

from rl.envs import RewardConfig, TicketToRideSelfPlayEnv

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_DIR = DEFAULT_PROJECT_ROOT / "src" / "map"
DEFAULT_TICKETS = DEFAULT_MAP_DIR / "tickets.csv"
DEFAULT_ROUTES = DEFAULT_MAP_DIR / "routes.csv"
DEFAULT_LOCATIONS = DEFAULT_MAP_DIR / "city_locations.json"


def load_graph(city_locations: Path, routes_file: Path):
    ttr_map = TicketToRideMap()
    ttr_map.load_graph(str(city_locations), str(routes_file))
    return ttr_map.get_graph()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RLlib self-play agents on Ticket to Ride")
    parser.add_argument("--iterations", type=int, default=300, help="Number of RLlib training iterations")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of rollout workers")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/rllib"), help="Directory for checkpoints and logs")
    parser.add_argument("--checkpoint-freq", type=int, default=20, help="Save checkpoint every N iterations")
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
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for PPO")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--framework", type=str, default="torch", help="RLlib backend framework (torch or tf)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def register_environment(args: argparse.Namespace, reward_config: RewardConfig):
    env_config = {
        "city_locations": str(args.city_locations),
        "routes_file": str(args.routes_file),
        "tickets_file": str(args.tickets_file),
        "reward_config": reward_config.__dict__,
    }

    def _env_creator(config):
        graph = load_graph(Path(config["city_locations"]), Path(config["routes_file"]))
        reward_cfg = RewardConfig(**config["reward_config"])
        return TicketToRideSelfPlayEnv(graph, config["tickets_file"], reward_config=reward_cfg)

    register_env("TicketToRideSelfPlay-v0", lambda config: _env_creator({**env_config, **config}))
    return env_config


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

    env_config = register_environment(args, reward_config)

    ray.init(ignore_reinit_error=True, log_to_driver=True)

    config = (
        PPOConfig()
        .environment("TicketToRideSelfPlay-v0", env_config=env_config)
        .framework(args.framework)
        .training(lr=args.lr, gamma=args.gamma)
        .rollouts(num_rollout_workers=args.num_workers, rollout_fragment_length=200)
        .resources(num_gpus=0)
        .debugging(seed=args.seed)
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
    )

    algo = config.build()

    for iteration in range(1, args.iterations + 1):
        result = algo.train()
        print(
            f"Iteration {iteration} - episode_reward_mean={result['episode_reward_mean']:.2f} "
            f"episode_len_mean={result['episode_len_mean']:.2f}"
        )

        if iteration % max(1, args.checkpoint_freq) == 0:
            checkpoint_path = algo.save(checkpoint_dir=str(log_dir))
            print(f"Checkpoint saved to {checkpoint_path}")

    evaluation = algo.evaluate()
    with open(log_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation results written to {log_dir / 'evaluation.json'}")

    algo.save(checkpoint_dir=str(log_dir / "final"))
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
