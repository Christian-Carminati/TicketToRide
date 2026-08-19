import datetime
import os
import sys
import uuid
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.agents.dqn_agent import DQNAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.api.replay_service import ReplayService
from src.api.schemas import ActionDTO, PlayerStateDTO, ReplayDetailDTO, ReplayFrameDTO
from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.env import TicketToRideEnv
from src.environment.observation import ObservationV1
from src.evaluation.evaluator import Evaluator
from src.evaluation.tournament import Tournament
from src.experiments.config import ExperimentConfig
from src.experiments.registry import ExperimentRecord, ExperimentRegistry
from src.game.game import Game
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.rl.dqn import MaskedDQNTrainer
from src.rl.ppo import MaskedPPOTrainer


def record_match_replay(agent_a, agent_b, replay_id: str, board, tickets, seed: int = 42) -> ReplayDetailDTO:
    """Simulate a game between two agents and record the full trajectory for replay viewer."""
    action_space = DiscreteActionSpace(board=board)
    masker = ActionMasker(action_space=action_space)
    encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=2)

    game = Game(board=board, tickets_deck=tickets, num_players=2, seed=seed)
    game.reset(seed=seed)

    frames: list[ReplayFrameDTO] = []
    step_idx = 0
    agents = [agent_a, agent_b]

    while not game.state.is_game_over and game.state.turn_number < 200:
        curr_idx = game.state.current_player_index
        curr_agent = agents[curr_idx]
        valid_actions = game.valid_actions()
        if not valid_actions:
            break

        # Snapshot of current state
        players_snap = []
        for p_idx, p in enumerate(game.state.players):
            players_snap.append(
                {
                    "player_id": p.id,
                    "name": p.name,
                    "score": p.score,
                    "trains_remaining": p.trains_remaining,
                    "cards_in_hand": {c.name: count for c, count in p.cards.items() if count > 0},
                    "tickets": [{"city_a": t.city_a, "city_b": t.city_b, "points": t.points, "completed": False} for t in p.tickets],
                    "claimed_route_ids": list(p.claimed_route_ids),
                    "color": "#3B82F6" if p_idx == 0 else "#EF4444",
                }
            )

        claimed_routes_map = {r.id: r.claimed_by for r in game.board.routes if r.claimed_by is not None}
        state_snap = {
            "turn_number": game.state.turn_number,
            "current_player_index": curr_idx,
            "players": players_snap,
            "claimed_routes": claimed_routes_map,
            "deck_size": len(game.state.train_deck),
            "visible_cards": [c.color.name for c in game.state.visible_cards],
        }

        # Select action
        chosen_action = curr_agent.act(game.state, valid_actions, board=game.board)
        score_before = game.state.players[curr_idx].score
        game.step(chosen_action)
        score_after = game.state.players[curr_idx].score
        reward = float(score_after - score_before)

        action_dto = ActionDTO(
            action_type=chosen_action.action_type.name,
            card_index=chosen_action.card_index,
            route_id=chosen_action.route_id,
            card_color=chosen_action.color_chosen.name if chosen_action.color_chosen else None,
            ticket_ids=list(chosen_action.ticket_ids) if chosen_action.ticket_ids else None,
        )

        frames.append(
            ReplayFrameDTO(
                step_index=step_idx,
                turn_number=game.state.turn_number,
                player_index=curr_idx,
                action=action_dto,
                reward=reward,
                state_snapshot=state_snap,
            )
        )
        step_idx += 1

    winner_idx = 0 if game.state.players[0].score >= game.state.players[1].score else 1
    replay = ReplayDetailDTO(
        replay_id=replay_id,
        map_name="mini",
        seed=seed,
        date=datetime.date.today().isoformat(),
        player_names=[agent_a.name, agent_b.name],
        total_steps=len(frames),
        winner_index=winner_idx,
        final_scores=[p.score for p in game.state.players],
        frames=frames,
    )
    return replay


def main():
    print("=" * 70)
    print("🚂 Ticket to Ride RL: Training & Multi-Opponent Evaluation Benchmark")
    print("=" * 70)

    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/replays", exist_ok=True)

    board, tickets = create_synthetic_mini_board()
    registry = ExperimentRegistry()
    replay_service = ReplayService()

    # -------------------------------------------------------------
    # 1. PPO Training
    # -------------------------------------------------------------
    print("\n[1/3] 🎯 Training Masked PPO Agent (20,000 steps)...")
    env_ppo = TicketToRideEnv(
        board=board,
        tickets_deck=tickets,
        opponent=RandomAgent(seed=42),
        num_players=2,
    )
    ppo_config = {
        "lr": 0.0005,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "rollout_steps": 128,
        "minibatch_size": 32,
        "num_epochs": 4,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    }
    ppo_trainer = MaskedPPOTrainer(env=env_ppo, config=ppo_config)

    total_timesteps_ppo = 20000
    current_steps = 0
    while current_steps < total_timesteps_ppo:
        ppo_trainer.collect_rollout()
        ppo_trainer.train_epoch()
        current_steps += ppo_trainer.rollout_steps
        if (current_steps // ppo_trainer.rollout_steps) % 20 == 0:
            print(f"  PPO Progress: {current_steps}/{total_timesteps_ppo} steps ({current_steps/total_timesteps_ppo*100:.0f}%)")

    ppo_ckpt_path = "experiments/checkpoints/ppo_mini_trained.pt"
    ppo_trainer.save(ppo_ckpt_path)
    print(f"  ✓ PPO Model Checkpoint saved to: {ppo_ckpt_path}")

    # -------------------------------------------------------------
    # 2. DQN Training
    # -------------------------------------------------------------
    print("\n[2/3] ⚡ Training Masked Double-DQN Agent (20,000 steps)...")
    env_dqn = TicketToRideEnv(
        board=board,
        tickets_deck=tickets,
        opponent=RandomAgent(seed=43),
        num_players=2,
    )
    dqn_config = {
        "lr": 0.0005,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 15000,
        "target_update_freq": 250,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 8000,
        "learning_starts": 200,
    }
    dqn_trainer = MaskedDQNTrainer(env=env_dqn, config=dqn_config)

    for step in range(1, 20001):
        dqn_trainer.step()
        dqn_trainer.train_step()
        if step % 5000 == 0:
            print(f"  DQN Progress: {step}/20000 steps ({step/20000*100:.0f}%) | ε = {dqn_trainer.get_epsilon():.3f}")

    dqn_ckpt_path = "experiments/checkpoints/dqn_mini_trained.pt"
    dqn_trainer.save(dqn_ckpt_path)
    print(f"  ✓ DQN Model Checkpoint saved to: {dqn_ckpt_path}")

    # -------------------------------------------------------------
    # 3. Multi-Opponent Evaluation & Tournament
    # -------------------------------------------------------------
    print("\n[3/3] 🏆 Evaluating Trained Agents vs Baselines (50 matches per pair)...")

    obs_dim = env_ppo.observation_space.shape[0]
    act_dim = int(env_ppo.action_space.n)

    trained_ppo = PPOAgent(
        name="PPO (Trained)",
        model_path=ppo_ckpt_path,
        input_dim=obs_dim,
        action_dim=act_dim,
        encoder=env_ppo.encoder,
        discrete_actions=env_ppo.discrete_actions,
    )

    trained_dqn = DQNAgent(
        name="DQN (Trained)",
        model_path=dqn_ckpt_path,
        input_dim=obs_dim,
        action_dim=act_dim,
        encoder=env_dqn.encoder,
        discrete_actions=env_dqn.discrete_actions,
    )

    random_agent = RandomAgent(name="RandomBot", seed=42)
    greedy_agent = GreedyAgent(name="GreedyBot")
    strategic_agent = StrategicAgent(name="StrategicBot")

    evaluator = Evaluator(board=board, tickets_deck=tickets, seed=100)

    opponents = [
        ("RandomBot", random_agent),
        ("GreedyBot", greedy_agent),
        ("StrategicBot", strategic_agent),
    ]

    ppo_metrics: dict[str, float] = {}
    dqn_metrics: dict[str, float] = {}

    print("\n📊 PPO Evaluation Results:")
    for name, opp in opponents:
        res = evaluator.evaluate_head_to_head(trained_ppo, opp, num_games=50)
        wr = res["agent_a_win_rate"]
        ppo_metrics[f"win_rate_vs_{name.lower()}"] = wr
        ppo_metrics[f"score_vs_{name.lower()}"] = res["agent_a_mean_score"]
        print(f"  • vs {name:15s}: Win Rate = {wr*100:5.1f}% | Mean Score = {res['agent_a_mean_score']:.1f} vs {res['agent_b_mean_score']:.1f}")

    print("\n📊 DQN Evaluation Results:")
    for name, opp in opponents:
        res = evaluator.evaluate_head_to_head(trained_dqn, opp, num_games=50)
        wr = res["agent_a_win_rate"]
        dqn_metrics[f"win_rate_vs_{name.lower()}"] = wr
        dqn_metrics[f"score_vs_{name.lower()}"] = res["agent_a_mean_score"]
        print(f"  • vs {name:15s}: Win Rate = {wr*100:5.1f}% | Mean Score = {res['agent_a_mean_score']:.1f} vs {res['agent_b_mean_score']:.1f}")

    # Head to Head: PPO vs DQN
    h2h = evaluator.evaluate_head_to_head(trained_ppo, trained_dqn, num_games=50)
    print(f"\n⚔️ PPO vs DQN Direct Matchup: PPO Win Rate = {h2h['agent_a_win_rate']*100:.1f}% | Mean Score: PPO {h2h['agent_a_mean_score']:.1f} - DQN {h2h['agent_b_mean_score']:.1f}")

    # -------------------------------------------------------------
    # 4. Round-Robin Tournament & Elo Ratings
    # -------------------------------------------------------------
    print("\n🏅 Round-Robin Tournament (5 Agents, 20 Games per matchup):")
    tournament_agents = [trained_ppo, trained_dqn, strategic_agent, greedy_agent, random_agent]
    tournament = Tournament(
        agents=tournament_agents,
        board=board,
        tickets_deck=tickets,
        games_per_pair=20,
    )
    t_results = tournament.run(seed=200)

    print("\nLeaderboard:")
    for rank, entry in enumerate(t_results["leaderboard"], 1):
        print(f"  {rank}. {entry['name']:20s} | Elo: {entry['elo']:.0f} | Win Rate: {entry['win_rate']*100:.1f}% | Wins: {entry['wins']}/{entry['total_games']}")

    # -------------------------------------------------------------
    # 5. Log Experiments to Registry
    # -------------------------------------------------------------
    exp_ppo = ExperimentRecord(
        experiment_id=f"ppo_mini_{uuid.uuid4().hex[:6]}",
        name="PPO Mini 20k Benchmark",
        seed=42,
        algorithm="ppo",
        env_version=1,
        reward_version=1,
        metrics=ppo_metrics,
    )
    registry.log_experiment(exp_ppo)

    exp_dqn = ExperimentRecord(
        experiment_id=f"dqn_mini_{uuid.uuid4().hex[:6]}",
        name="Double-DQN Mini 20k Benchmark",
        seed=43,
        algorithm="dqn",
        env_version=1,
        reward_version=1,
        metrics=dqn_metrics,
    )
    registry.log_experiment(exp_dqn)

    # -------------------------------------------------------------
    # 6. Generate Replay Recordings
    # -------------------------------------------------------------
    print("\n🎞️ Generating Match Replays for Web Lab Viewer...")
    replay1 = record_match_replay(trained_ppo, greedy_agent, "replay_ppo_vs_greedy_mini", board, tickets, seed=501)
    replay_service.save_replay(replay1)

    replay2 = record_match_replay(trained_ppo, strategic_agent, "replay_ppo_vs_strategic_mini", board, tickets, seed=502)
    replay_service.save_replay(replay2)

    replay3 = record_match_replay(trained_dqn, greedy_agent, "replay_dqn_vs_greedy_mini", board, tickets, seed=503)
    replay_service.save_replay(replay3)

    replay4 = record_match_replay(trained_ppo, trained_dqn, "replay_ppo_vs_dqn_mini", board, tickets, seed=504)
    replay_service.save_replay(replay4)

    print("  ✓ Saved 4 rich match replays into experiments/replays/ (accessible in Replay Player UI)")
    print("\n🎉 Training and Evaluation Benchmark Completed Successfully!")


if __name__ == "__main__":
    main()
