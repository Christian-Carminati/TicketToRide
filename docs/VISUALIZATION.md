# Game Visualization Guide

This guide explains how to use the visualization system to watch two agents play Ticket to Ride against each other.

## Overview

The visualization system provides:
- **Real-time game state visualization**: See routes being claimed, player hands, scores
- **Interactive viewing**: Watch games play out step by step
- **Game recording**: Save game history for later replay
- **Agent comparison**: Visualize different agent strategies

## Quick Start

### Basic Usage

```python
from game import Game
from map import TicketToRideMap
from visualization import GameViewer

# Load map and create game
ttr_map = TicketToRideMap()
ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
game = Game(number_of_players=2, graph=ttr_map.get_graph())
game.setup()

# Create viewer
viewer = GameViewer(game)
viewer.setup_figure()

# Update and show
viewer.update()
viewer.show()
```

### Running Demo

```bash
python -m src.visualization.demo_agents
```

This will:
1. Create a 2-player game
2. Set up visualization
3. Run a game between a greedy agent and a random agent
4. Display the game state in real-time

## Components

### GameViewer

Main visualization class that displays:
- **Map**: Network graph with claimed routes colored by player
- **Player Information**: Scores, trains remaining, cards in hand, routes claimed
- **Game Status**: Current turn, final round indicator, game over status

### GameRecorder

Records game state at each step for later replay.

### visualize_agent_game()

High-level function to run a game between two agent functions with visualization.

## Visualization Features

### Map Display

- **Unclaimed routes**: Shown in light gray with dashed lines
- **Claimed routes**: Colored by player (red, blue, green, yellow, black)
- **Route thickness**: Proportional to route length
- **Route labels**: Show length (1-6) on each route
- **Cities**: Labeled with city names

### Information Panel

Shows for each player:
- **Color**: Player's color
- **Score**: Current points
- **Trains**: Remaining train pieces (out of 45)
- **Routes**: Number of routes claimed
- **Cards**: Total cards in hand + breakdown by color
- **Tickets**: Number of destination tickets

### Status Indicators

- **Current Turn**: Highlighted player color
- **Game Status**: IN PROGRESS / FINAL ROUND / GAME OVER
- **Winner**: Displayed at end of game

## Example: Custom Agent Visualization

```python
from game import Game, Color
from map import TicketToRideMap
from visualization import GameViewer
import matplotlib.pyplot as plt

def my_agent(game: Game, player_index: int) -> dict:
    """Your custom agent logic."""
    player = game.players[player_index]
    
    # Your strategy here
    # Return action dict:
    # {'type': 'claim_route', 'city1': ..., 'city2': ..., 'color': ..., 'color_to_use': ...}
    # OR
    # {'type': 'draw_cards', 'face_up_index': None or 0-4}
    pass

# Setup
ttr_map = TicketToRideMap()
ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
game = Game(number_of_players=2, graph=ttr_map.get_graph())
game.setup()

viewer = GameViewer(game)
viewer.setup_figure()

# Game loop
while not game.is_game_over:
    player_idx = game.current_player_index
    action = my_agent(game, player_idx)
    
    # Execute action
    if action['type'] == 'claim_route':
        game.claim_route_action(
            action['city1'], action['city2'],
            action['color'], action['color_to_use']
        )
    else:
        game.draw_card_action(action['face_up_index'])
    
    # Update visualization
    viewer.update()
    plt.pause(0.5)  # Delay for visibility

viewer.show(block=True)
```

## Saving Frames

To save visualization frames for creating videos:

```python
viewer = GameViewer(game)
viewer.setup_figure()

for step in range(num_steps):
    # ... game logic ...
    viewer.update()
    viewer.save_frame(f"frames/step_{step:04d}.png")
```

Then create video with ffmpeg:
```bash
ffmpeg -r 2 -i frames/step_%04d.png -c:v libx264 -pix_fmt yuv420p game.mp4
```

## Integration with RL Agents

To visualize trained RL agents:

```python
from stable_baselines3 import PPO
from rl.envs import SingleAgentTicketToRideEnv

# Load trained model
model = PPO.load("path/to/model")

# Create environment
env = SingleAgentTicketToRideEnv(graph, tickets_file)

# Convert RL actions to game actions
def rl_agent_to_game_action(obs, model):
    action, _ = model.predict(obs, deterministic=True)
    # Convert action index to route claim
    route = env.routes[action]
    return {
        'type': 'claim_route',
        'city1': route[0],
        'city2': route[1],
        'color': route[3],
        'color_to_use': route[3]  # Simplified
    }
```

## Troubleshooting

### Map Not Displaying

- Ensure graph has position attributes: `graph.nodes[node]['pos']`
- Check that `city_locations.json` is loaded correctly

### Visualization Too Slow

- Reduce `plt.pause()` delay
- Disable frame saving
- Use smaller figure size

### Cards Not Showing

- Ensure game is properly set up with `game.setup()`
- Check that players have cards in hand

## Advanced Usage

### Custom Layout

Modify `GameViewer.setup_figure()` to change layout:

```python
# Different grid layout
gs = self.fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
self.ax_map = self.fig.add_subplot(gs[0, 0])
self.ax_info = self.fig.add_subplot(gs[0, 1])
```

### Multiple Views

Create multiple viewers for different perspectives:

```python
viewer1 = GameViewer(game)
viewer2 = GameViewer(game)  # Same game, different window
```

### Recording Games

```python
from visualization import GameRecorder

recorder = GameRecorder(game)

# During game
recorder.record_state("Player 1 claimed route")

# Replay later
recorder.replay(viewer, delay=1.0)
```

## Limitations

Current implementation:
- ✅ Shows game state clearly
- ✅ Displays player information
- ✅ Real-time updates
- ❌ No interactive controls (pause/play/step)
- ❌ No animation between states
- ❌ Limited to 2D visualization

Future enhancements:
- Interactive controls
- 3D visualization
- Animation between moves
- Statistical overlays
- Replay controls

