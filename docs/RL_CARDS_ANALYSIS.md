# RL Training and Card Mechanics Analysis

## Current State Analysis

### ❌ Problem: Cards Are NOT Considered in RL Environments

After analyzing the codebase, **the current RL environments do NOT properly model the card mechanics** of Ticket to Ride. This is a significant limitation.

### Current RL Environment Implementation

#### `SingleAgentTicketToRideEnv` (Gymnasium)

**Observation Space:**
```python
# Observation: [agent_routes, opponent_routes, agent_trains, opponent_trains,
#               agent_claims_norm, opponent_claims_norm]
obs_len = 2 * self.num_routes + 4
```

**What's Missing:**
- ❌ No information about cards in hand
- ❌ No information about available face-up cards
- ❌ No information about deck state
- ❌ No card color requirements for routes

**Action Space:**
```python
self.action_space = spaces.Discrete(self.num_routes)
```
- Actions are simply route indices
- No consideration of whether player has required cards
- Routes can be claimed without card validation

#### `TicketToRideSelfPlayEnv` (PettingZoo)

Similar limitations:
- Same observation space (no cards)
- Same action space (route indices only)
- No card mechanics

### What This Means

1. **Unrealistic Training**: Agents learn to claim routes without considering card availability
2. **Invalid Strategies**: Agents may learn strategies that are impossible in real gameplay
3. **Missing Game Mechanics**: The core resource management aspect (collecting cards) is ignored

## Comparison: Full Game vs RL Environment

### Full Game (`game.py`)

✅ **Complete Implementation:**
- Players have hands with cards (`Hand` class)
- Deck management (`Deck` class)
- Face-up cards (5 visible cards)
- Card drawing mechanics
- Route claiming requires matching cards
- Color requirements (specific color or grey for any)

### RL Environment (`rl/envs.py`)

❌ **Simplified Implementation:**
- No cards
- Routes can be claimed if:
  - City pair not already claimed
  - Enough trains remaining
- No card color validation

## Impact on Training

### Current Training Behavior

Agents learn to:
- Select routes based on points/train efficiency
- Consider connectivity for tickets
- Avoid claiming routes already taken

But they **cannot learn**:
- When to draw cards vs claim routes
- Which cards to collect
- Strategic card management
- Timing of route claims based on card availability

### Realistic Gameplay Requirements

In real Ticket to Ride:
1. Players start with 4 random cards
2. Each turn: draw 2 cards OR claim 1 route
3. To claim a route, must have matching color cards
4. Grey routes can use any color
5. Jolly cards can substitute any color
6. Face-up cards visible to all players

## Recommendations

### Option 1: Enhance Current RL Environment (Recommended)

Add card mechanics to existing environments:

```python
# Enhanced observation space
obs_len = (
    2 * self.num_routes +      # Agent/opponent routes
    4 +                        # Trains remaining
    9 +                        # Cards in hand (one-hot per color)
    5 +                        # Face-up cards
    1                          # Deck size (normalized)
)
```

**New Action Space:**
```python
# Multi-discrete action space
self.action_space = spaces.MultiDiscrete([
    self.num_routes + 1,  # Route to claim (or skip)
    6,                    # Card to draw from face-up (0-4) or deck (5)
    6,                    # Second card to draw
])
```

### Option 2: Use Full Game Class

Integrate `Game` class directly into RL environment:

```python
class FullGameRLEnv(gym.Env):
    def __init__(self, ...):
        self.game = Game(number_of_players=2, graph=graph)
        self.game.setup()
    
    def step(self, action):
        # Action: (action_type, action_params)
        # action_type: 'draw_cards' or 'claim_route'
        if action[0] == 'draw_cards':
            self.game.draw_card_action(...)
        elif action[0] == 'claim_route':
            self.game.claim_route_action(...)
```

### Option 3: Hybrid Approach

Keep simplified environment for initial training, add full game for fine-tuning.

## Implementation Plan

### Phase 1: Analysis ✅
- [x] Document current state
- [x] Identify gaps

### Phase 2: Enhanced Environment
- [ ] Add card state to observations
- [ ] Expand action space to include card drawing
- [ ] Add card validation for route claiming
- [ ] Update reward function to consider card efficiency

### Phase 3: Full Game Integration
- [ ] Create `FullGameRLEnv` using `Game` class
- [ ] Implement proper action encoding
- [ ] Add observation normalization

### Phase 4: Training Updates
- [ ] Update training scripts
- [ ] Adjust hyperparameters for larger action space
- [ ] Test with card-aware agents

## Code Changes Needed

### 1. Observation Space Enhancement

```python
def _get_state(self) -> np.ndarray:
    # Current routes (existing)
    agent_routes = np.zeros(self.num_routes)
    for route in self.agent_routes:
        idx = self.pair_to_idx[tuple(sorted((route[0], route[1])))]
        agent_routes[idx] = 1.0
    
    # NEW: Cards in hand (one-hot encoding per color)
    agent_cards = np.zeros(9)  # 8 colors + jolly
    for color, count in self.agent_hand.cards.items():
        color_idx = self._color_to_idx(color)
        agent_cards[color_idx] = min(count / 12.0, 1.0)  # Normalize
    
    # NEW: Face-up cards
    face_up = np.zeros(5)
    for i, card in enumerate(self.face_up_cards[:5]):
        if card:
            face_up[i] = self._color_to_idx(card) / 8.0
    
    # Combine
    return np.concatenate([
        agent_routes,
        opponent_routes,
        [self.trains_agent / 45.0],
        [self.trains_opponent / 45.0],
        agent_cards,
        face_up
    ])
```

### 2. Action Space Enhancement

```python
# Multi-discrete: [route_action, card_action_1, card_action_2]
self.action_space = spaces.MultiDiscrete([
    self.num_routes + 1,  # 0 = skip, 1-N = claim route
    6,                    # 0-4 = face-up card, 5 = deck
    6                     # Second card (if first was face-up)
])
```

### 3. Step Function Updates

```python
def step(self, action):
    route_action, card1, card2 = action
    
    if route_action > 0:
        # Try to claim route
        route_idx = route_action - 1
        if self._can_claim_route(route_idx):
            self._claim_route(route_idx)
    else:
        # Draw cards
        self._draw_cards(card1, card2)
    
    # ... rest of step logic
```

## Conclusion

The current RL implementation is **significantly simplified** and does not capture the strategic depth of Ticket to Ride. Adding card mechanics would:

1. ✅ Make training more realistic
2. ✅ Enable learning of card management strategies
3. ✅ Better match real gameplay
4. ✅ Improve agent performance in actual games

However, this comes with:
- ⚠️ Larger observation space
- ⚠️ More complex action space
- ⚠️ Longer training times
- ⚠️ More complex reward shaping

## ✅ Implementation Complete!

**Status**: Full game environment with card mechanics has been implemented!

See **[RL_FULL_GAME.md](RL_FULL_GAME.md)** for:
- Complete documentation of the new environment
- Usage examples
- Training scripts
- Comparison with simplified environment

**New Files**:
- `src/rl/full_game_env.py`: Full game environments with cards
- `src/rl/train_full_game.py`: Training script for full game

**Key Features Implemented**:
1. ✅ Card state in observations
2. ✅ Card drawing actions
3. ✅ Route validation with cards
4. ✅ Full Game class integration

