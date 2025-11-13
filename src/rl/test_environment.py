"""
Quick test to verify the environment is working correctly.
"""
from pathlib import Path
import numpy as np

from map import TicketToRideMap
from rl.full_game_env import FullGameSingleAgentEnv, RewardConfig
from rl.opponents import build_opponent
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


def main():
    print("="*70)
    print("🧪 ENVIRONMENT TEST")
    print("="*70)
    
    # Setup
    map_dir = Path("map")
    city_locations = map_dir / "city_locations.json"
    routes_file = map_dir / "routes.csv"
    tickets_file = map_dir / "tickets.csv"
    
    print("\n1️⃣ Loading map...")
    ttr_map = TicketToRideMap()
    ttr_map.load_graph(str(city_locations), str(routes_file))
    graph = ttr_map.get_graph()
    print(f"   ✅ Loaded {len(graph.nodes)} cities, {len(graph.edges)} routes")
    
    print("\n2️⃣ Creating environment...")
    reward_config = RewardConfig(
        invalid_action_penalty=10.0,
        efficiency_weight=0.5,
        connectivity_bonus=5.0,
        ticket_weight=0.05,
        final_score_scale=0.1,
        card_draw_bonus=0.1,
    )
    
    opponent = build_opponent("greedy", graph=graph, tickets_file=str(tickets_file))
    base_env = FullGameSingleAgentEnv(
        graph, str(tickets_file), opponent_policy=opponent, reward_config=reward_config
    )
    env = TimeLimit(base_env, max_episode_steps=800)
    env = Monitor(env)
    
    print(f"   ✅ Environment created")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    print("\n3️⃣ Resetting environment...")
    obs, info = env.reset()
    print(f"   ✅ Reset complete")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation (first 10 values): {obs[:10]}")
    
    print("\n4️⃣ Testing 10 random actions...")
    print("-"*70)
    
    game = base_env.game
    
    for step in range(10):
        # Get valid actions
        valid_actions = base_env.get_valid_actions(for_agent=True)
        
        # Choose a random valid action
        if valid_actions:
            action = np.random.choice(valid_actions)
        else:
            print(f"   ⚠️ No valid actions at step {step}!")
            break
        
        # Get game state BEFORE action
        p0_before = game.players[0]
        p1_before = game.players[1]
        trains_before = p0_before.trains_left
        routes_before = len(p0_before.routes)
        cards_before = sum(p0_before.hand.cards.values())
        score_before = p0_before.score
        
        # Execute action
        obs_new, reward, done, truncated, info = env.step(action)
        
        # Get game state AFTER action
        p0_after = game.players[0]
        p1_after = game.players[1]
        trains_after = p0_after.trains_left
        routes_after = len(p0_after.routes)
        cards_after = sum(p0_after.hand.cards.values())
        score_after = p0_after.score
        
        # Check if state changed
        state_changed = (
            trains_before != trains_after or
            routes_before != routes_after or
            cards_before != cards_after or
            score_before != score_after
        )
        
        # Determine action type
        action_type = "ROUTE" if action < base_env.num_routes else "CARD/TICKET"
        
        print(f"\nStep {step}: Action {action:3d} ({action_type}) - Reward: {reward:6.3f}")
        print(f"   Valid actions available: {len(valid_actions)}")
        print(f"   State changed: {'✅ YES' if state_changed else '❌ NO'}")
        print(f"   Agent:")
        print(f"      Trains: {trains_before} → {trains_after} (Δ {trains_after-trains_before})")
        print(f"      Routes: {routes_before} → {routes_after} (Δ {routes_after-routes_before})")
        print(f"      Cards:  {cards_before} → {cards_after} (Δ {cards_after-cards_before})")
        print(f"      Score:  {score_before} → {score_after} (Δ {score_after-score_before})")
        print(f"   Opponent:")
        print(f"      Trains: {p1_before.trains_left} → {p1_after.trains_left}")
        print(f"      Routes: {len(p1_before.routes)} → {len(p1_after.routes)}")
        
        # Check if observation changed
        obs_changed = not np.allclose(obs, obs_new)
        print(f"   Observation changed: {'✅ YES' if obs_changed else '❌ NO'}")
        
        if not state_changed and not obs_changed:
            print(f"   🚨 PROBLEM: Neither state nor observation changed!")
        
        obs = obs_new
        
        if done or truncated:
            print(f"\n   Episode ended (done={done}, truncated={truncated})")
            break
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    if game.players[0].trains_left == 45 and len(game.players[0].routes) == 0:
        print("❌ CRITICAL ISSUE: Game state NEVER changed!")
        print("\n   Possible problems:")
        print("   1. env.step() not calling the game correctly")
        print("   2. Actions not being executed")
        print("   3. Game loop not advancing")
        print("\n   Check: rl/full_game_env.py step() method")
    else:
        print("✅ Environment appears to be working!")
        print(f"   Agent claimed {len(game.players[0].routes)} routes")
        print(f"   Agent has {sum(game.players[0].hand.cards.values())} cards")
        print(f"   Agent used {45 - game.players[0].trains_left} trains")
    
    print("="*70)


if __name__ == "__main__":
    main()