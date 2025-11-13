"""
Game visualization system for Ticket to Ride.

This module provides visualization tools to watch two agents play against each other,
showing the game state, claimed routes, player hands, and scores in real-time.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

from game import Game, Player, Route, Color
from map import TicketToRideMap


class GameViewer:
    """
    Visualizes a Ticket to Ride game between two agents.
    
    Shows:
    - Map with claimed routes (colored by player)
    - Player hands (cards)
    - Scores and trains remaining
    - Current turn indicator
    - Destination tickets
    """

    def __init__(self, game: Game, figsize: Tuple[int, int] = (20, 12)) -> None:
        """
        Initialize the game viewer.
        
        Args:
            game: Game instance to visualize
            figsize: Figure size (width, height)
        """
        self.game = game
        self.figsize = figsize
        self.fig: Optional[plt.Figure] = None
        self.ax_map: Optional[plt.Axes] = None
        self.ax_info: Optional[plt.Axes] = None
        
        # Color mapping for players
        self.player_colors_map = {
            Color.RED: '#FF4444',
            Color.BLUE: '#4444FF',
            Color.GREEN: '#44FF44',
            Color.YELLOW: '#FFFF44',
            Color.BLACK: '#444444',
        }
        
        # Get graph positions
        self.pos = nx.get_node_attributes(game.graph, 'pos')
        if not self.pos:
            # Fallback: use spring layout
            self.pos = nx.spring_layout(game.graph, k=2, iterations=50)
    
    def setup_figure(self) -> None:
        """Create and configure the matplotlib figure."""
        self.fig = plt.figure(figsize=self.figsize, facecolor='white')
        
        # Create grid layout: map on left, info on right
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Map takes 2 rows, 2 columns
        self.ax_map = self.fig.add_subplot(gs[:, :2])
        self.ax_map.set_title("Ticket to Ride - Game Map", fontsize=16, fontweight='bold')
        self.ax_map.axis('off')
        
        # Info panels
        self.ax_info = self.fig.add_subplot(gs[:, 2])
        self.ax_info.axis('off')
    
    def draw_map(self) -> None:
        """Draw the game map with claimed routes."""
        if self.ax_map is None:
            return
        
        self.ax_map.clear()
        self.ax_map.set_title("Ticket to Ride - Game Map", fontsize=16, fontweight='bold')
        self.ax_map.axis('off')
        
        # Draw all available routes (unclaimed) in light gray
        for u, v, data in self.game.graph.edges(data=True):
            if (u, v) not in self._get_claimed_edges():
                color_code = str(data.get('color', 'X')).upper()[0]
                edge_color = self._color_code_to_hex(color_code)
                self.ax_map.plot(
                    [self.pos[u][0], self.pos[v][0]],
                    [self.pos[u][1], self.pos[v][1]],
                    color=edge_color,
                    alpha=0.2,
                    linewidth=1,
                    linestyle='--'
                )
        
        # Draw claimed routes (colored by player)
        for player in self.game.players:
            player_color = self.player_colors_map.get(player.color, '#888888')
            for route in player.routes:
                u, v = route.city1, route.city2
                if u in self.pos and v in self.pos:
                    self.ax_map.plot(
                        [self.pos[u][0], self.pos[v][0]],
                        [self.pos[u][1], self.pos[v][1]],
                        color=player_color,
                        linewidth=route.length * 2,
                        alpha=0.8,
                        solid_capstyle='round'
                    )
                    # Add route length label
                    mid_x = (self.pos[u][0] + self.pos[v][0]) / 2
                    mid_y = (self.pos[u][1] + self.pos[v][1]) / 2
                    self.ax_map.text(
                        mid_x, mid_y,
                        str(route.length),
                        fontsize=8,
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                    )
        
        # Draw cities
        for city, (x, y) in self.pos.items():
            self.ax_map.plot(x, y, 'o', color='black', markersize=8, zorder=5)
            self.ax_map.text(
                x, y + 0.02,
                city,
                fontsize=7,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )
    
    def draw_info_panel(self) -> None:
        """Draw information panel with player stats."""
        if self.ax_info is None:
            return
        
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        y_start = 0.95
        y_spacing = 0.15
        
        # Title
        self.ax_info.text(
            0.5, y_start,
            "Game Status",
            fontsize=14,
            fontweight='bold',
            ha='center',
            transform=self.ax_info.transAxes
        )
        
        y_pos = y_start - 0.1
        
        # Current turn indicator
        current_player = self.game.current_player()
        turn_text = f"Current Turn: {current_player.color.value.upper()}"
        self.ax_info.text(
            0.5, y_pos,
            turn_text,
            fontsize=12,
            fontweight='bold',
            ha='center',
            color=self.player_colors_map.get(current_player.color, 'black'),
            transform=self.ax_info.transAxes
        )
        
        y_pos -= y_spacing * 1.5
        
        # Player information
        for i, player in enumerate(self.game.players):
            player_color = self.player_colors_map.get(player.color, '#888888')
            
            # Player header
            header = f"Player {i+1}: {player.color.value.upper()}"
            self.ax_info.text(
                0.05, y_pos,
                header,
                fontsize=11,
                fontweight='bold',
                color=player_color,
                transform=self.ax_info.transAxes
            )
            
            y_pos -= 0.05
            
            # Score
            score_text = f"  Score: {player.score}"
            self.ax_info.text(
                0.05, y_pos,
                score_text,
                fontsize=9,
                transform=self.ax_info.transAxes
            )
            
            y_pos -= 0.04
            
            # Trains remaining
            trains_text = f"  Trains: {player.trains_left}/45"
            self.ax_info.text(
                0.05, y_pos,
                trains_text,
                fontsize=9,
                transform=self.ax_info.transAxes
            )
            
            y_pos -= 0.04
            
            # Routes claimed
            routes_text = f"  Routes: {len(player.routes)}"
            self.ax_info.text(
                0.05, y_pos,
                routes_text,
                fontsize=9,
                transform=self.ax_info.transAxes
            )
            
            y_pos -= 0.04
            
            # Cards in hand
            total_cards = sum(player.hand.cards.values())
            cards_text = f"  Cards: {total_cards}"
            self.ax_info.text(
                0.05, y_pos,
                cards_text,
                fontsize=9,
                transform=self.ax_info.transAxes
            )
            
            # Show card breakdown
            y_pos -= 0.03
            card_details = []
            for color, count in sorted(player.hand.cards.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    card_details.append(f"{color.value[:3]}:{count}")
            if card_details:
                cards_detail = "  " + ", ".join(card_details[:5])  # Show max 5
                if len(card_details) > 5:
                    cards_detail += "..."
                self.ax_info.text(
                    0.05, y_pos,
                    cards_detail,
                    fontsize=7,
                    style='italic',
                    transform=self.ax_info.transAxes
                )
            
            y_pos -= 0.05
            
            # Destination tickets
            tickets_text = f"  Tickets: {len(player.destination_tickets)}"
            self.ax_info.text(
                0.05, y_pos,
                tickets_text,
                fontsize=9,
                transform=self.ax_info.transAxes
            )
            
            y_pos -= y_spacing
        
        # Game status
        if self.game.is_game_over:
            status_text = "GAME OVER"
            status_color = 'red'
        elif self.game.is_final_round:
            status_text = "FINAL ROUND"
            status_color = 'orange'
        else:
            status_text = "IN PROGRESS"
            status_color = 'green'
        
        self.ax_info.text(
            0.5, 0.05,
            status_text,
            fontsize=12,
            fontweight='bold',
            ha='center',
            color=status_color,
            transform=self.ax_info.transAxes
        )
    
    def _get_claimed_edges(self) -> set:
        """Get set of claimed edges."""
        claimed = set()
        for player in self.game.players:
            for route in player.routes:
                claimed.add((route.city1, route.city2))
                claimed.add((route.city2, route.city1))
        return claimed
    
    def _color_code_to_hex(self, code: str) -> str:
        """Convert color code to hex color."""
        color_map = {
            'R': '#FF0000', 'B': '#0000FF', 'G': '#00FF00',
            'Y': '#FFFF00', 'O': '#FF8800', 'K': '#000000',
            'W': '#C0C0C0', 'P': '#8800FF', 'X': '#808080'
        }
        return color_map.get(code, '#808080')
    
    def update(self) -> None:
        """Update the visualization."""
        self.draw_map()
        self.draw_info_panel()
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def show(self, block: bool = True) -> None:
        """Display the visualization."""
        if self.fig is None:
            self.setup_figure()
            self.update()
        plt.show(block=block)
    
    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        if self.fig is None:
            self.setup_figure()
            self.update()
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
    
    def close(self) -> None:
        """Close the figure."""
        if self.fig:
            plt.close(self.fig)


class GameRecorder:
    """
    Records a game between two agents and allows replay visualization.
    """
    
    def __init__(self, game: Game) -> None:
        """Initialize game recorder."""
        self.game = game
        self.history: List[Dict] = []
        self.current_step = 0
    
    def record_state(self, action_description: str = "") -> None:
        """Record current game state."""
        state = {
            'step': len(self.history),
            'current_player': self.game.current_player_index,
            'action': action_description,
            'players': []
        }
        
        for player in self.game.players:
            player_state = {
                'color': player.color.value,
                'score': player.score,
                'trains_left': player.trains_left,
                'routes': [(r.city1, r.city2, r.length, r.color.value) for r in player.routes],
                'cards': dict((k.value, v) for k, v in player.hand.cards.items()),
                'tickets': [(t.city1, t.city2, t.points) for t in player.destination_tickets]
            }
            state['players'].append(player_state)
        
        state['game_over'] = self.game.is_game_over
        state['final_round'] = self.game.is_final_round
        
        self.history.append(state)
    
    def replay(self, viewer: GameViewer, delay: float = 1.0) -> None:
        """
        Replay recorded game with visualization.
        
        Args:
            viewer: GameViewer instance
            delay: Delay between steps in seconds
        """
        # This would require restoring game state, which is complex
        # For now, just show the final state
        viewer.show()


def visualize_agent_game(
    game: Game,
    agent1_action_fn,
    agent2_action_fn,
    step_delay: float = 1.0,
    save_frames: bool = False,
    output_dir: Optional[Path] = None
) -> None:
    """
    Visualize a game between two agent functions.
    
    Args:
        game: Game instance
        agent1_action_fn: Function that takes (game, player_index) and returns action
        agent2_action_fn: Function that takes (game, player_index) and returns action
        step_delay: Delay between moves in seconds
        save_frames: Whether to save frames to files
        output_dir: Directory to save frames (if save_frames=True)
    """
    viewer = GameViewer(game)
    viewer.setup_figure()
    
    if save_frames and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    # Initial state
    viewer.update()
    if save_frames:
        viewer.save_frame(str(output_dir / f"frame_{frame_count:04d}.png"))
    plt.pause(step_delay)
    
    # Game loop
    while not game.is_game_over:
        current_player = game.current_player()
        player_idx = game.current_player_index
        
        # Get action from appropriate agent
        if player_idx == 0:
            action = agent1_action_fn(game, player_idx)
        else:
            action = agent2_action_fn(game, player_idx)
        
        # Execute action (simplified - would need proper action execution)
        # For now, this is a placeholder
        
        viewer.update()
        frame_count += 1
        if save_frames:
            viewer.save_frame(str(output_dir / f"frame_{frame_count:04d}.png"))
        
        plt.pause(step_delay)
    
    # Final state
    viewer.update()
    if save_frames:
        viewer.save_frame(str(output_dir / f"frame_{frame_count:04d}.png"))
    
    # Show winner
    winner, scores = game.calculate_winner()
    print(f"\n{'='*60}")
    print(f"GAME OVER - Winner: {winner.color.value.upper()}")
    print(f"{'='*60}")
    for player in game.players:
        print(f"{player.color.value.upper()}: {scores[player.color]} points")
    
    viewer.show(block=True)

