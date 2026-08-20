"""Comprehensive PPO Benchmark & Ablation Study Runner."""

import json
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.evaluator import Evaluator
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.rl.ppo import MaskedPPOTrainer


class PPOBenchmarkRunner:
    """Automated benchmark runner for PPO models against standard baseline agents with Ablation Study."""

    def __init__(
        self,
        board_type: str = "mini",
        games_per_opponent: int = 50,
        total_training_steps: int = 2000,
        training_steps: int | None = None,
        output_json: str | None = None,
        output_md: str | None = None,
        seed: int = 42,
    ) -> None:
        self.board_type = board_type
        self.games_per_opponent = games_per_opponent
        self.total_training_steps = training_steps if training_steps is not None else total_training_steps
        self.output_json = output_json
        self.output_md = output_md
        self.seed = seed


        if board_type == "mini":
            self.board, self.tickets = create_synthetic_mini_board()
        else:
            self.board, self.tickets = load_usa_board()

    def train_agent(
        self,
        config: dict[str, Any] | None = None,
        steps: int | None = None,
        seed: int | None = None,
    ) -> tuple[PPOAgent, dict[str, Any]]:
        """Train a PPO agent with the specified configuration and timesteps."""
        train_seed = self.seed if seed is None else seed
        train_steps = self.total_training_steps if steps is None else steps

        env = TicketToRideEnv(
            board=self.board,
            tickets_deck=self.tickets,
            opponent=RandomAgent(seed=train_seed),
        )

        default_cfg = {
            "lr": 3e-4,
            "anneal_lr": True,
            "clip_vloss": True,
            "target_kl": 0.03,
            "rollout_steps": 128 if self.board_type == "mini" else 256,
            "num_epochs": 4,
            "minibatch_size": 32 if self.board_type == "mini" else 64,
            "orthogonal_init": True,
        }
        if config:
            default_cfg.update(config)

        trainer = MaskedPPOTrainer(env=env, config=default_cfg)
        train_summary = trainer.train(total_timesteps=train_steps)

        agent = PPOAgent(
            name="PPO",
            actor_critic=trainer.actor_critic,
            encoder=env.encoder,
            discrete_actions=env.discrete_actions,
            device=trainer.device,
        )
        return agent, train_summary

    def evaluate_opponents(
        self,
        agent: BaseAgent,
        games: int | None = None,
        seed: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate an agent head-to-head against Random, Greedy, and Strategic baselines."""
        eval_games = self.games_per_opponent if games is None else games
        eval_seed = self.seed if seed is None else seed

        opponents = {
            "random": RandomAgent(seed=eval_seed + 100),
            "greedy": GreedyAgent(),
            "strategic": StrategicAgent(),
        }

        results: dict[str, dict[str, float]] = {}
        evaluator = Evaluator(board=self.board, tickets_deck=self.tickets, seed=eval_seed)

        for name, opp in opponents.items():
            opp_results = evaluator.evaluate(agent_a=agent, agent_b=opp, num_games=eval_games, seed=eval_seed)
            m_a = opp_results[agent.name]
            m_b = opp_results[opp.name]

            results[name] = {
                "win_rate": float(m_a.win_rate),
                "agent_mean_score": float(m_a.avg_score),
                "opponent_mean_score": float(m_b.avg_score),
                "score_differential": float(m_a.avg_score_diff),
                "ticket_completion_rate": float(m_a.ticket_completion_rate),
                "draw_rate": float(m_a.draws / max(1, eval_games)),
            }

        return results

    def run_ablation_study(
        self,
        ablation_steps: int = 500,
        ablation_games: int = 20,
    ) -> dict[str, dict[str, Any]]:
        """Run an automated ablation study measuring the impact of CleanRL components."""
        variants: dict[str, dict[str, Any]] = {
            "ppo_full": {"orthogonal_init": True, "clip_vloss": True, "anneal_lr": True},
            "ppo_no_ortho": {"orthogonal_init": False, "clip_vloss": True, "anneal_lr": True},
            "ppo_no_vf_clip": {"orthogonal_init": True, "clip_vloss": False, "anneal_lr": True},
            "ppo_no_lr_anneal": {"orthogonal_init": True, "clip_vloss": True, "anneal_lr": False},
        }

        ablation_results: dict[str, dict[str, Any]] = {}
        for var_name, cfg in variants.items():
            agent, train_summary = self.train_agent(
                config=cfg,
                steps=ablation_steps,
                seed=self.seed,
            )
            eval_res = self.evaluate_opponents(
                agent=agent,
                games=ablation_games,
                seed=self.seed + 500,
            )
            ablation_results[var_name] = {
                "config": cfg,
                "train_summary": train_summary,
                "vs_random_win_rate": eval_res["random"]["win_rate"],
                "vs_random_score_diff": eval_res["random"]["score_differential"],
                "vs_greedy_win_rate": eval_res["greedy"]["win_rate"],
                "vs_strategic_win_rate": eval_res["strategic"]["win_rate"],
            }

        return ablation_results

    def generate_markdown_report(self, results: dict[str, Any]) -> str:
        """Generate a formatted markdown report from benchmark and ablation results."""
        lines = [
            f"# PPO Benchmark & Ablation Report ({results['board_type'].upper()} Map)",
            "",
            f"- **Map**: `{results['board_type']}`",
            f"- **Training Steps**: `{results['training_steps']}`",
            f"- **Games per Opponent**: `{results['games_per_opponent']}`",
            "",
            "## 1. Performance vs Baseline Opponents",
            "",
            "| Opponent | Win Rate (%) | PPO Avg Score | Opponent Avg Score | Score Diff | Ticket Completion (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for opp, data in results["opponents"].items():
            lines.append(
                f"| **{opp.capitalize()}** | {data['win_rate'] * 100:.1f}% | {data['agent_mean_score']:.1f} | "
                f"{data['opponent_mean_score']:.1f} | {data['score_differential']:+.1f} | "
                f"{data['ticket_completion_rate'] * 100:.1f}% |"
            )

        if "ablation" in results:
            lines.extend([
                "",
                "## 2. CleanRL Ablation Study",
                "",
                "| Variant | Vs Random Win% | Vs Random Diff | Vs Greedy Win% | Vs Strategic Win% | Description |",
                "| :--- | :---: | :---: | :---: | :---: | :--- |",
            ])
            for var_name, data in results["ablation"].items():
                desc = {
                    "ppo_full": "Full CleanRL (Ortho + VF Clip + LR Anneal)",
                    "ppo_no_ortho": "Standard initialization (No Orthogonal)",
                    "ppo_no_vf_clip": "Unclipped value loss",
                    "ppo_no_lr_anneal": "Constant learning rate",
                }.get(var_name, var_name)

                lines.append(
                    f"| `{var_name}` | {data['vs_random_win_rate'] * 100:.1f}% | {data['vs_random_score_diff']:+.1f} | "
                    f"{data['vs_greedy_win_rate'] * 100:.1f}% | {data['vs_strategic_win_rate'] * 100:.1f}% | {desc} |"
                )

        lines.append("")
        return "\n".join(lines)

    def run_benchmark(
        self,
        run_ablation: bool = True,
        ablation_steps: int = 500,
        ablation_games: int = 20,
    ) -> dict[str, Any]:
        """Execute full benchmark, run ablation study if requested, and export reports."""
        agent, train_summary = self.train_agent(steps=self.total_training_steps)
        opp_results = self.evaluate_opponents(agent=agent, games=self.games_per_opponent)

        results: dict[str, Any] = {
            "board_type": self.board_type,
            "training_steps": self.total_training_steps,
            "games_per_opponent": self.games_per_opponent,
            "train_summary": train_summary,
            "opponents": opp_results,
        }

        if run_ablation:
            results["ablation"] = self.run_ablation_study(
                ablation_steps=ablation_steps,
                ablation_games=ablation_games,
            )

        if self.output_json:
            out_json = Path(self.output_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        if self.output_md:
            out_md = Path(self.output_md)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            md_content = self.generate_markdown_report(results)
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(md_content)

        return results
