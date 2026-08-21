"""Headless Self-Play training entrypoint."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.env import TicketToRideEnv
from src.game.maps import load_usa_board
from src.rl.self_play import SelfPlayPPOTrainer, SelfPlayRecurrentPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a self-play agent headless")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    parser.add_argument("--snapshot-interval", type=int, default=2000, help="Interval between snapshots")
    parser.add_argument("--strategy", type=str, default="latest_biased", help="Matchmaking strategy")
    parser.add_argument("--baseline-mix", type=float, default=0.15, help="Baseline mix-in rate")
    parser.add_argument("--recurrent", action="store_true", help="Use recurrent LSTM architecture")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    board, tickets = load_usa_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, seed=args.seed)

    cfg = {
        "rollout_steps": 256,
        "num_epochs": 2,
        "snapshot_interval": args.snapshot_interval,
        "sampling_strategy": args.strategy,
        "baseline_mix_rate": args.baseline_mix,
    }

    if args.recurrent:
        cfg["seq_len"] = 8
        cfg["minibatch_chunks"] = 4
        trainer = SelfPlayRecurrentPPOTrainer(env=env, config=cfg, seed=args.seed)
    else:
        trainer = SelfPlayPPOTrainer(env=env, config=cfg, seed=args.seed)

    print(f"Starting Self-Play training ({'Recurrent' if args.recurrent else 'MLP'}) for {args.timesteps} timesteps...")
    trainer.train(total_timesteps=args.timesteps)
    print(f"Self-play training completed. Total snapshots in pool: {trainer.pool.size}")


if __name__ == "__main__":
    main()
