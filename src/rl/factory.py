"""Unified TrainerFactory for instantiating RL trainers across experiments and API services."""

from typing import Any

from src.rl.alphazero_trainer import AlphaZeroTrainer
from src.rl.dqn import MaskedDQNTrainer
from src.rl.lstm_ppo import MaskedRecurrentPPOTrainer
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.ppo import MaskedPPOTrainer
from src.rl.self_play import SelfPlayPPOTrainer, SelfPlayRecurrentPPOTrainer


class TrainerFactory:
    """Instantiates algorithm trainers with standardized signatures and validated hyperparameters."""

    SUPPORTED_ALGORITHMS: list[str] = [
        "dqn",
        "ppo",
        "recurrent_ppo",
        "lstm_ppo",
        "self_play_ppo",
        "self_play_recurrent_ppo",
        "alphazero",
        "neural_mcts",
    ]

    @classmethod
    def create(
        cls,
        algorithm: str,
        env: Any,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        algo = algorithm.strip().lower()
        cfg = config or {}

        if algo == "dqn":
            return MaskedDQNTrainer(env=env, config=cfg)

        elif algo == "ppo":
            if cfg.get("self_play", False) or kwargs.get("self_play", False):
                return SelfPlayPPOTrainer(env=env, config=cfg, seed=seed)
            return MaskedPPOTrainer(env=env, config=cfg)

        elif algo in ("recurrent_ppo", "lstm_ppo"):
            if cfg.get("self_play", False) or kwargs.get("self_play", False):
                return SelfPlayRecurrentPPOTrainer(env=env, config=cfg, seed=seed)
            return MaskedRecurrentPPOTrainer(env=env, config=cfg, seed=seed)

        elif algo == "self_play_ppo":
            return SelfPlayPPOTrainer(env=env, config=cfg, seed=seed)

        elif algo == "self_play_recurrent_ppo":
            return SelfPlayRecurrentPPOTrainer(env=env, config=cfg, seed=seed)

        elif algo in ("alphazero", "neural_mcts"):
            obs_dim = (
                env.observation_space.shape[0]
                if getattr(env, "observation_space", None) is not None
                and env.observation_space.shape
                else 180
            )
            act_dim = (
                int(env.action_space.n) if getattr(env, "action_space", None) is not None else 150
            )
            num_simulations = kwargs.get("num_simulations", cfg.get("num_simulations", 30))
            pv_net = PolicyValueNetwork(
                obs_dim=obs_dim,
                action_dim=act_dim,
                hidden_dim=cfg.get("hidden_dim", 64),
                num_res_blocks=cfg.get("num_res_blocks", 1),
            )
            board = getattr(env, "board", None)
            tickets = getattr(env, "initial_tickets", None)
            return AlphaZeroTrainer(
                net=pv_net,
                num_simulations=num_simulations,
                c_puct=cfg.get("c_puct", 1.5),
                batch_size=cfg.get("batch_size", 16),
                lr=cfg.get("lr", 1e-3),
                device=cfg.get("device", "cpu"),
                board=board,
                tickets=tickets,
            )

        else:
            raise ValueError(
                f"Unsupported algorithm: '{algorithm}'. Available algorithms: {cls.SUPPORTED_ALGORITHMS}"
            )
