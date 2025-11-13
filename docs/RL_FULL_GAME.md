# Full Game RL Environment with Card Mechanics

## Overview

The `FullGameSingleAgentEnv` and `FullGameSelfPlayEnv` are new RL environments that include **complete card mechanics** from the Ticket to Ride game. Unlike the simplified environments in `rl/envs.py`, these environments:

✅ **Include card management**: Players have hands with train cards  
✅ **Model the deck**: Face-up cards and deck state  
✅ **Validate route claims**: Routes can only be claimed if player has required cards  
✅ **Support card drawing**: Agents can draw cards as actions  
✅ **Use full Game class**: Complete game logic integration

## Key Features

### Observation Space

The observation includes:

1. **Agent routes** (num_routes): Binary vector of claimed routes
2. **Opponent routes** (num_routes): Binary vector of opponent's routes
3. **Agent trains** (1): Normalized remaining trains (0-1)
4. **Opponent trains** (1): Normalized remaining trains
5. **Agent cards** (9): Normalized card counts per color (RED, BLUE, GREEN, YELLOW, ORANGE, BLACK, SILVER, PURPLE, JOLLY)
6. **Opponent cards** (9): Partial info (total cards, not exact distribution)
7. **Face-up cards** (5 × 9): One-hot encoding of face-up cards (5 positions × 9 colors)
8. **Deck size** (1): Normalized remaining deck cards

**Total observation size**: `2 × num_routes + 2 + 18 + 45 + 1`

### Action Space

Actions are encoded as:

- **0 to num_routes-1**: Claim a specific route (if valid)
- **num_routes to num_routes+12**: Draw cards (various combinations)
  - `num_routes + 0`: Draw 2 from deck
  - `num_routes + 1-5`: Draw face-up card at index (action-1), then deck
  - `num_routes + 6-10`: Draw deck, then face-up card at index (action-6)
  - `num_routes + 11`: Try to draw jolly (if available)
  - `num_routes + 12`: Skip turn (penalty)
- **num_routes + 13**: Draw destination tickets

**Total action space**: `num_routes + 14`

### Route Claiming Validation

Routes can only be claimed if:
1. Route is not already claimed
2. Player has enough trains remaining
3. **Player has required cards** (new!)
   - For colored routes: Must have route color + jollys
   - For grey routes: Can use any color + jollys

### Card Drawing

Agents can draw cards in various ways:
- From deck (random)
- From face-up cards (visible to all)
- Combinations of both
- Special action to target jolly cards

## Usage

### Single Agent Training

```python
from rl.full_game_env import FullGameSingleAgentEnv, RewardConfig
from rl.opponents import build_opponent
from map import TicketToRideMap

# Load map
ttr_map = TicketToRideMap()
ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
graph = ttr_map.get_graph()

# Create opponent
opponent = build_opponent("greedy", graph=graph, tickets_file="src/map/tickets.csv")

# Create environment
reward_config = RewardConfig(
    invalid_action_penalty=10.0,
    efficiency_weight=0.5,
    connectivity_bonus=5.0,
    ticket_weight=0.05,
    final_score_scale=0.1,
    card_draw_bonus=0.1,  # Bonus for drawing cards
)

env = FullGameSingleAgentEnv(
    graph,
    "src/map/tickets.csv",
    opponent_policy=opponent,
    reward_config=reward_config
)

# Train with Stable Baselines3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
```

### Self-Play Training

```python
from rl.full_game_env import FullGameSelfPlayEnv

env = FullGameSelfPlayEnv(graph, "src/map/tickets.csv")

# Use with RLlib for multi-agent training
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig().multi_agent(
    policies={
        "shared_policy": PolicySpec(),
    },
    policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
)

algo = config.build(env="TicketToRideFullGame-v0")
```

### Training Script

Use the provided training script:

```bash
python -m src.rl.train_full_game \
  --algorithm ppo \
  --timesteps 200000 \
  --opponent greedy \
  --log-dir runs/full_game
```

## Comparison with Simplified Environment

| Feature | Simplified (`envs.py`) | Full Game (`full_game_env.py`) |
|---------|------------------------|--------------------------------|
| **Cards** | ❌ Not modeled | ✅ Complete card management |
| **Deck** | ❌ Not modeled | ✅ Deck and face-up cards |
| **Route Validation** | Trains only | ✅ Trains + Cards |
| **Card Drawing** | ❌ Not available | ✅ Multiple draw actions |
| **Observation Size** | `2×routes + 4` | `2×routes + 66` |
| **Action Size** | `routes` | `routes + 14` |
| **Realism** | Low | High |
| **Training Complexity** | Low | Higher |

## Reward Structure

Rewards include:

1. **Route points**: Base points for claimed route
2. **Efficiency bonus**: Points per train ratio
3. **Ticket points**: Bonus for completed destination tickets
4. **Connectivity bonus**: Bonus for longest path
5. **Card draw bonus**: Small bonus for drawing cards (encourages collection)
6. **Final score**: Scaled difference in final scores

## Valid Actions

The environment provides `get_valid_actions()` that returns only actions that are:
- **Route claims**: Routes that can be claimed (available, enough trains, **enough cards**)
- **Card draws**: Always valid (but may have different outcomes)
- **Ticket draws**: Always valid

## Example: Testing Trained Agent

```python
from stable_baselines3 import PPO
from rl.full_game_env import FullGameSingleAgentEnv
from map import TicketToRideMap

# Load model
model = PPO.load("runs/full_game/best_model/best_model")

# Create environment
ttr_map = TicketToRideMap()
ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
env = FullGameSingleAgentEnv(ttr_map.get_graph(), "src/map/tickets.csv")

# Test
obs, _ = env.reset()
total_reward = 0
steps = 0

while steps < 1000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    
    if terminated or truncated:
        break

print(f"Final scores: Agent={info['agent_score']}, Opponent={info['opponent_score']}")
env.render()  # Show detailed game state
```

## Migration from Simplified Environment

If you have models trained on the simplified environment:

1. **Observation space is larger**: Retrain from scratch
2. **Action space is larger**: New actions for card drawing
3. **Action encoding changed**: Route indices are the same, but new actions added
4. **Validation is stricter**: Invalid actions more common initially

**Recommendation**: Start with simplified environment for quick training, then fine-tune on full game environment.

## Performance Considerations

- **Larger observation space**: May require larger neural networks
- **More actions**: Longer training time
- **Card validation**: More computation per step
- **Realistic gameplay**: Better transfer to actual games

## Tips for Training

1. **Start with card_draw_bonus**: Encourage agents to collect cards early
2. **Use curriculum learning**: Start with simpler opponents, increase difficulty
3. **Monitor card usage**: Check if agents learn to collect cards strategically
4. **Compare with simplified**: Train both and compare strategies

## Future Enhancements

- [ ] Action masking for invalid route claims
- [ ] Partial observability options
- [ ] Card counting strategies
- [ ] Multi-step planning rewards

