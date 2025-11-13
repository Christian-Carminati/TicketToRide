"""
Visualize a trained SB3 agent playing the full game environment.
FIXED VERSION with better error handling and debugging.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from map import TicketToRideMap
from rl.full_game_env import FullGameSingleAgentEnv, RewardConfig
from rl.opponents import build_opponent
from visualization.game_viewer import GameViewer
from gymnasium.wrappers import TimeLimit


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
    max_episode_steps: int,
) -> FullGameSingleAgentEnv:
    graph = load_graph(city_locations, routes_file)
    opponent = build_opponent(opponent_name, graph=graph, tickets_file=str(tickets_file))
    base_env = FullGameSingleAgentEnv(
        graph, str(tickets_file), opponent_policy=opponent, reward_config=reward_config
    )
    env = TimeLimit(base_env, max_episode_steps=max_episode_steps)
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a trained agent on the full game")
    parser.add_argument("--model-path", type=Path, default=Path("runs/full_game/final_model.zip"), help="Path to PPO/DQN model zip")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--opponent", type=str, default="greedy", choices=["random", "greedy", "heuristic"])
    parser.add_argument("--map-dir", type=Path, default=Path("map"))
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--step-delay", type=float, default=0.5, help="Seconds between frames")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step debug info")
    parser.add_argument("--no-viz", action="store_true", help="Run without visualization (for debugging)")
    args = parser.parse_args()

    # Check if model exists
    if not args.model_path.exists():
        print(f"❌ ERROR: Model not found at {args.model_path}")
        print(f"   Did you train a model first?")
        print(f"   Run: python -m rl.train_full_game --timesteps 200000")
        sys.exit(1)

    # Files
    city_locations = args.map_dir / "city_locations.json"
    routes_file = args.map_dir / "routes.csv"
    tickets_file = args.map_dir / "tickets.csv"

    # Check map files exist
    for f in [city_locations, routes_file, tickets_file]:
        if not f.exists():
            print(f"❌ ERROR: Map file not found: {f}")
            sys.exit(1)

    print("="*70)
    print("🎮 Ticket to Ride - Agent Evaluation")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Opponent: {args.opponent}")
    print(f"Max steps: {args.max_episode_steps}")
    print("="*70)

    reward_config = RewardConfig(
        invalid_action_penalty=10.0,
        efficiency_weight=0.5,
        connectivity_bonus=5.0,
        ticket_weight=0.05,
        final_score_scale=0.1,
        card_draw_bonus=0.1,
    )

    print("\n📦 Creating environment...")
    try:
        env = make_env(
            city_locations=city_locations,
            routes_file=routes_file,
            tickets_file=tickets_file,
            opponent_name=args.opponent,
            reward_config=reward_config,
            max_episode_steps=args.max_episode_steps,
        )
        print("✅ Environment created")
    except Exception as e:
        print(f"❌ ERROR creating environment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n🤖 Loading {args.algorithm.upper()} model...")
    try:
        if args.algorithm == "ppo":
            model = PPO.load(str(args.model_path), env=env)
        else:
            model = DQN.load(str(args.model_path), env=env)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n🎲 Starting game...")
    print("-"*70)

    try:
        obs, info = env.reset()
        print(f"✅ Game initialized")
        if args.verbose:
            print(f"   Initial observation shape: {obs.shape}")
            print(f"   Info: {info}")
    except Exception as e:
        print(f"❌ ERROR during reset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Setup visualization AFTER reset so we bind to the fresh Game instance
    viewer = None
    base_env = env.unwrapped
    if not args.no_viz:
        try:
            print("\n🎨 Setting up visualization...")
            game = base_env.game
            viewer = GameViewer(game)
            viewer.setup_figure()
            
            import matplotlib.pyplot as plt
            plt.ion()
            viewer.show(block=False)
            print("✅ Visualization ready")
        except Exception as e:
            print(f"⚠️  WARNING: Could not setup visualization: {e}")
            print("   Continuing without visualization...")
            args.no_viz = True

    done = False
    truncated = False
    step_idx = 0
    total_reward = 0.0
    invalid_action_count = 0
    
    try:
        # base_env already set above
        
        while not (done or truncated):
            # Get action from model
            try:
                action, _ = model.predict(obs, deterministic=True)
                # Handle different return types from predict
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:  # Scalar array
                        action = int(action.item())
                    else:  # Regular array
                        action = int(action[0])
                elif isinstance(action, list):
                    action = int(action[0])
                else:
                    action = int(action)
            except Exception as e:
                print(f"❌ ERROR predicting action: {e}")
                break

            # Validate action and fallback if invalid
            try:
                valid_actions = base_env.get_valid_actions(for_agent=True)
                if action not in valid_actions:
                    invalid_action_count += 1
                    if args.verbose:
                        print(f"   ⚠️  Invalid action {action}, choosing from {len(valid_actions)} valid actions")
                    
                    if valid_actions:
                        # Prefer route claiming actions
                        num_routes = getattr(base_env, "num_routes", 0)
                        route_actions = [a for a in valid_actions if 0 <= a < num_routes]
                        action = int(route_actions[0] if route_actions else valid_actions[0])
                    else:
                        print("   ❌ No valid actions available!")
                        break
            except Exception as e:
                if args.verbose:
                    print(f"   ⚠️  Could not validate action: {e}")

            # Execute action
            try:
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if args.verbose:
                    try:
                        num_routes = getattr(base_env, "num_routes", 0)
                        action_desc = (
                            f"claim route #{action}" if 0 <= action < num_routes
                            else f"draw/ticket action #{action}"
                        )
                        game = base_env.game
                        p0 = game.players[0]
                        p1 = game.players[1]
                        print(f"[Step {step_idx:3d}] action={action:3d} ({action_desc:20s}) "
                              f"reward={reward:6.2f} total={total_reward:7.2f} | "
                              f"Agent: trains={p0.trains_left:2d} routes={len(p0.routes):2d} | "
                              f"Opp: trains={p1.trains_left:2d} routes={len(p1.routes):2d}")
                    except Exception:
                        print(f"[Step {step_idx:3d}] action={action} reward={reward:.2f}")
                elif step_idx % 10 == 0:
                    print(f"   Step {step_idx}: reward={reward:.2f}, total={total_reward:.2f}")
                
            except Exception as e:
                print(f"❌ ERROR during step {step_idx}: {e}")
                import traceback
                traceback.print_exc()
                break

            # Update visualization
            if viewer is not None:
                try:
                    viewer.update()
                    import matplotlib.pyplot as plt
                    plt.pause(args.step_delay)
                    
                    # Check if window was closed
                    if not plt.fignum_exists(viewer.fig.number):
                        print("\n⚠️  Visualization window closed by user")
                        break
                except Exception as e:
                    if args.verbose:
                        print(f"   ⚠️  Visualization update failed: {e}")

            step_idx += 1

            # Safety check for infinite loops
            if step_idx > args.max_episode_steps * 2:
                print(f"\n⚠️  Safety limit reached ({step_idx} steps), stopping...")
                break

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ ERROR during game: {e}")
        import traceback
        traceback.print_exc()

    # Final statistics
    print("\n" + "="*70)
    print("📊 GAME RESULTS")
    print("="*70)
    print(f"Steps played: {step_idx}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Invalid actions: {invalid_action_count} ({100*invalid_action_count/max(step_idx,1):.1f}%)")
    print(f"Game completed: {done}")
    print(f"Truncated: {truncated}")
    
    try:
        if "agent_score" in info:
            print(f"\n🏆 Final Scores:")
            print(f"   Agent:    {info['agent_score']} points")
            print(f"   Opponent: {info['opponent_score']} points")
            if info['agent_score'] > info['opponent_score']:
                print(f"   🎉 AGENT WINS by {info['agent_score'] - info['opponent_score']} points!")
            elif info['agent_score'] < info['opponent_score']:
                print(f"   😞 Opponent wins by {info['opponent_score'] - info['agent_score']} points")
            else:
                print(f"   🤝 TIE!")
    except Exception:
        pass
    
    print("="*70)

    # Keep visualization open
    if viewer is not None:
        try:
            print("\n👁️  Final visualization - Close window to exit")
            viewer.update()
            import matplotlib.pyplot as plt
            plt.ioff()
            viewer.show(block=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()