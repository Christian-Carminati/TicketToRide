"""Game state and logic for Ticket to Ride."""
from __future__ import annotations

import enum
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from map import TicketToRideMap

class Color(enum.Enum):
    RED = 'red'
    BLUE = 'blue'
    GREEN = 'green'
    YELLOW = 'yellow'
    ORANGE = 'orange'
    BLACK = 'black'
    SILVER = 'silver'
    PURPLE = 'purple'
    JOLLY = 'jolly'
    GREY = 'grey'
    
class MoveType(enum.Enum):
    DRAW_CARDS = 'draw_cards'
    CLAIM_ROUTE = 'claim_route'
    DRAW_TICKETS = 'draw_tickets'
    
    
@dataclass
class Route:
    city1: str
    city2: str
    length: int
    color: Color
    owner: Optional[Color] = None
    
    def can_claim(self, player_hand: Dict[Color, int], color_to_use: Color) -> bool:
        """
        Check if the player has enough cards of the given color to claim the route.

        :param player_hand: The player's hand of cards.
        :type player_hand: Dict[Color, int]
        :param color_to_use: The color of cards to use to claim the route.
        :type color_to_use: Color
        :return: Whether the player has enough cards to claim the route.
        :rtype: bool
        """
        if self.owner is not None:
            return False
        
        if self.color == Color.GREY:
            total = player_hand.get(color_to_use, 0) + player_hand.get(Color.JOLLY, 0)
            return total >= self.length
        else:
            total = player_hand.get(self.color, 0) + player_hand.get(Color.JOLLY, 0)
            return total >= self.length

@dataclass
class DestinationTicket:
    city1: str
    city2: str
    points: int
    
    def is_completed(self, player_routes: List[Route]) -> bool:
        """
        Check if a player has completed the destination ticket by claiming routes that connect the two cities.

        :param player_routes: A list of Route objects representing the routes claimed by the player.
        :type player_routes: List[Route]
        :return: Whether the player has completed the destination ticket.
        :rtype: bool
        """
        graph = nx.Graph()
        for route in player_routes:
            graph.add_edge(route.city1, route.city2)
        
        if self.city1 not in graph or self.city2 not in graph:
            return False
        
        return nx.has_path(graph, self.city1, self.city2)


class Deck:
    def __init__(self, cards: Optional[Dict[Color, int]] = None) -> None:
        self.cards: Dict[Color, int] = dict(cards) if cards else {}

    @staticmethod
    def full_deck() -> 'Deck':
        """
        Return a full deck of Ticket To Ride cards.
        
        The deck contains 12 cards of each color (red, blue, green, yellow, orange, black, silver, purple) and 14 jolly cards.
        
        :return: A Deck object containing a full deck of Ticket To Ride cards.
        :rtype: Deck
        """
        # Initialize a deck with all the colors
        cards = {
            Color.RED: 12,
            Color.BLUE: 12,
            Color.GREEN: 12,
            Color.YELLOW: 12,
            Color.ORANGE: 12,
            Color.BLACK: 12,
            Color.SILVER: 12,
            Color.PURPLE: 12,
            Color.JOLLY: 14
        }
        # Create a Deck object with the cards
        deck = Deck(cards)
        return deck


    def draw(self) -> Color | None:
        """
        Draw a random card from the deck (so remove it).

        :return: A Color object representing the card drawn from the deck.
        :rtype: Color
        """

        colors: List[Color] = []
        weights: List[int] = []

        for color, count in self.cards.items():
            if count > 0:
                colors.append(color)
                weights.append(count)
                
        if not colors:  # Double check
            return None
        drawn_card: Color = random.choices(colors, weights=weights, k=1)[0]
        
        self.cards[drawn_card] -= 1
        
        return drawn_card

    def add_card(self, cards: List[Color]) -> None:
        for card in cards:
            if card in self.cards:
                self.cards[card] += 1
            else:
                self.cards[card] = 1

    def is_empty(self) -> bool:
        return all(count == 0 for count in self.cards.values())
    
class Hand:
    def __init__(self) -> None:
        self.cards: Dict[Color, int] = {}              
    def add_card(self, card: Color) -> None:
        """
        Add a card to the player's hand.

        :param card: A Color object representing the card to add.
        :type card: Color
        :return: None
        """
        if card in self.cards:
            self.cards[card] += 1
        else:
            self.cards[card] = 1      
            
    def remove_card(self, card: Color, count: int = 1) -> bool:
        """
        Remove a card from the player's hand.

        :param card: A Color object representing the card to remove.
        :type card: Color
        :param count: The number of cards to remove.
        :type count: int
        :return: True if cards were removed, False if not enough cards
        :rtype: bool
        """
        if card in self.cards and self.cards[card] >= count:
            self.cards[card] -= count
            if self.cards[card] == 0:
                del self.cards[card]
            return True
        return False
        
    def can_claim_route(self, route: Route, color_to_use: Color) -> bool:
        """
        Check if the player can claim the given route with their current hand.
        
        Delegates to Route.can_claim to avoid code duplication.

        Args:
            route: A Route object representing the route to claim
            color_to_use: The color of cards to use to claim the route
            
        Returns:
            Whether the player can claim the route
        """
        return route.can_claim(self.cards, color_to_use)
        
class Player:
    def __init__(self, color: Color) -> None:
        self.color: Color = color
        self.hand: Hand = Hand()
        self.trains_left: int = 45
        self.score: int = 0
        self.routes: List[Route] = []
        self.destination_tickets: List[DestinationTicket] = []
        
    def claim_route(self, route: Route, color_to_use: Color) -> bool:
        """
        Claim a route on the board.

        :param route: The route to claim.
        :type route: Route
        :param color_to_use: The color of cards to use to claim the route.
        :type color_to_use: Color
        :return: Whether the player successfully claimed the route.
        :rtype: bool
        """
        # Check if player can claim the route
        if not self.hand.can_claim_route(route, color_to_use):
            return False
        
        # Determine which color to use
        if route.color == Color.GREY:
            needed_color: Color = color_to_use
        else:
            needed_color: Color = route.color
        
        # Calculate how many cards of each type to remove
        available: int = self.hand.cards.get(needed_color, 0)
        jolly_needed: int = max(0, route.length - available)
        color_needed: int = route.length - jolly_needed
        
        # Remove the color cards
        if color_needed > 0:
            if not self.hand.remove_card(needed_color, color_needed):
                return False
        
        # Remove the jolly cards
        if jolly_needed > 0:
            if not self.hand.remove_card(Color.JOLLY, jolly_needed):
                # Rollback if we can't remove jolly cards
                if color_needed > 0:
                    # Add back the color cards we removed
                    for _ in range(color_needed):
                        self.hand.add_card(needed_color)
                return False
        
        # Successfully removed cards, now claim the route
        route.owner = self.color
        self.routes.append(route)
        self.trains_left -= route.length
        
        # Add points for the route
        points_table: Dict[int, int] = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
        self.score += points_table.get(route.length, 0)
        
        return True
    
class GameState:
    def __init__(self) -> None:
        """
        Initialize a GameState object.

        :return: None
        :rtype: None
        """
        self.face_up_cards: List[Color] = []  # Face-up cards
        self.deck: Deck = Deck()  # The deck of cards
        self.discard_pile: List[Color] = []  # The discard pile
        self.destination_ticket_deck: List[DestinationTicket] = []  # The deck of destination tickets

    def setup_deck(self) -> None:
        """
        Initialize the deck of cards and the face-up cards.
        
        :return: None
        :rtype: None
        """
        self.deck: Deck = Deck.full_deck()
        self.face_up_cards: List[Color] = []
        for _ in range(5):
            card: Optional[Color] = self.deck.draw()
            if card:
                self.face_up_cards.append(card)
        self._check_up_locomotives()

    def _check_up_locomotives(self) -> None:
        """
        Check if there are three or more jolly cards in the face-up cards.
        If so, discard all face-up cards and draw new ones.
        
        :return: None
        """
        jolly_count: int = self.face_up_cards.count(Color.JOLLY)
        if jolly_count >= 3:
            self.discard_pile.extend(self.face_up_cards)
            self.face_up_cards: List[Color] = []
            for _ in range(5):
                card: Optional[Color] = self.deck.draw()
                if card:
                    self.face_up_cards.append(card)
            if self.deck.is_empty():
                return
            self._check_up_locomotives()

    def draw_face_up_card(self, index: int) -> Optional[Color]:
        """
        Draw a face-up card from the specified index.
        
        :param index: The index of the face-up card to draw (0-4).
        :type index: int
        :return: The drawn Color card, or None if the index is invalid.
        :rtype: Optional[Color]
        """
        if index < 0 or index >= len(self.face_up_cards):
            return None
        
        drawn_card: Color = self.face_up_cards.pop(index)
        
        new_card: Optional[Color] = self.deck.draw()
        if new_card:
            self.face_up_cards.insert(index, new_card)
        
        self._check_up_locomotives()
        
        return drawn_card
    
    def draw_deck_card(self) -> Optional[Color]:
        """
        Draw a card from the deck.
        
        :return: The drawn Color card, or None if the deck is empty.
        :rtype: Optional[Color]
        """
        """Draw a card from the deck."""
        if self.deck.is_empty():
            # Rimescola le carte scartate nel mazzo
            if len(self.discard_pile) == 0:
                return None
            
            # Sposta le carte scartate nel mazzo
            discard_counts = Counter(self.discard_pile)
            for color, count in discard_counts.items():
                self.deck.cards[color] = self.deck.cards.get(color, 0) + count
            self.discard_pile.clear()
            
        
        return self.deck.draw()
    
        
    def setup_destination_tickets(self, tickets: List[DestinationTicket]) -> None:
        """
        Setup the destination ticket deck.
        
        :param tickets: A list of DestinationTicket objects to use as the deck.
        :type tickets: List[DestinationTicket]
        :return: None
        """
        self.destination_ticket_deck = tickets.copy()
        random.shuffle(self.destination_ticket_deck)
        
    def draw_destination_tickets(self, count: int = 3) -> List[DestinationTicket]:
        """
        Draw destination tickets from the deck.
        
        :param count: The number of destination tickets to draw.
        :type count: int
        :return: A list of drawn DestinationTicket objects.
        :rtype: List[DestinationTicket]
        """
        drawn_tickets: List[DestinationTicket] = []
        for _ in range(count):
            if self.destination_ticket_deck:
                ticket: DestinationTicket = self.destination_ticket_deck.pop()
                drawn_tickets.append(ticket)
        return drawn_tickets
    
    def return_destination_tickets(self, tickets: List[DestinationTicket]) -> None:
        self.destination_ticket_deck.extend(tickets)

class RouteManager:
    def __init__(self, graph: nx.MultiGraph) -> None:
        self.routes: List[Route] = []
        self._load_route_from_graph(graph)

    def _load_route_from_graph(self, graph: nx.MultiGraph) -> None:
        """
        Load routes from the graph into the route manager.

        :param graph: The graph to load routes from.
        :type graph: nx.MultiGraph
        :return: None
        """
        for u, v, data in graph.edges(data=True):
            color_code: str = str(data.get('color', 'X')).upper()[0]
            color_map: Dict[str, Color] = {
                'R': Color.RED, 'B': Color.BLUE, 'G': Color.GREEN,
                'Y': Color.YELLOW, 'O': Color.ORANGE, 'K': Color.BLACK,
                'W': Color.SILVER, 'P': Color.PURPLE, 'X': Color.GREY
            }
            color: Color = color_map.get(color_code, Color.GREY)
            route: Route = Route(
                city1=u,
                city2=v,
                length=data.get('weight', 1),
                color=color
            )
            self.routes.append(route)
            
    def get_available_routes(self, number_of_players: int) -> List[Route]:
        """
        Get a list of available (unclaimed) routes.
        
        If the number of players is 2, only return routes that have not been claimed by the same player.

        :param number_of_players: The number of players in the game.
        :type number_of_players: int
        :return: A list of available Route objects.
        :rtype: List[Route]
        """
        available_routes: List[Route] = [route for route in self.routes if route.owner is None]
        if number_of_players == 2:
            filtered_routes: List[Route] = []
            seen_pairs: set = set()
            for route in available_routes:
                pair = tuple(sorted((route.city1, route.city2)))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    filtered_routes.append(route)
            return filtered_routes
        return available_routes
    
    def get_route(
        self, 
        city1: str, 
        city2: str, 
        color: Color
    ) -> Optional[Route]:
        """
        Get a route object from the route manager if it exists.

        :param city1: The first city of the route.
        :type city1: str
        :param city2: The second city of the route.
        :type city2: str
        :param color: The color of the route.
        :type color: Color
        :return: The Route object if it exists, otherwise None.
        :rtype: Optional[Route]
        """
        for route in self.routes:
            if ((route.city1 == city1 and route.city2 == city2) or
                (route.city1 == city2 and route.city2 == city1)):
                if route.color == color and route.owner is None:
                    return route
        return None

class ScoreCalculator:
    @staticmethod
    def calculate_final_scores(players: List[Player]) -> Dict[Color, int]:
        """
        Calculate the final scores for all players, including destination ticket points.

        :param players: A list of Player objects.
        :type players: List[Player]
        :return: A dictionary mapping player colors to their final scores.
        :rtype: Dict[Color, int]
        """
        final_scores: Dict[Color, int] = {}
        for player in players:
            total_score: int = player.score
            for ticket in player.destination_tickets:
                if ticket.is_completed(player.routes):
                    total_score += ticket.points
                else:
                    total_score -= ticket.points
            final_scores[player.color] = total_score
        return final_scores
    
    @staticmethod
    def longest_route_bonus(players: List[Player]) -> Color | None:
        """
        Determine the player with the longest continuous route and award them a bonus.

        :param players: A list of Player objects.
        :type players: List[Player]
        :return: The Color of the player with the longest route.
        :rtype: Color
        """
        longest_length: int = 0
        longest_player: Optional[Color] = None
        
        for player in players:
            graph = nx.Graph()
            for route in player.routes:
                graph.add_edge(route.city1, route.city2, weight=route.length)
            
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                length = sum(data['weight'] for u, v, data in subgraph.edges(data=True))
                if length > longest_length:
                    longest_length = length
                    longest_player = player.color
                    
        return longest_player


class Game:
    def __init__(self, number_of_players: int, graph: nx.MultiGraph) -> None:
        self.number_of_players: int = number_of_players
        
        if number_of_players < 2 or number_of_players > 5:
            raise ValueError("Number of players must be between 2 and 5.")
        
        self.graph: nx.MultiGraph = graph
        self.players: List[Player] = []
        self.route_manager: RouteManager = RouteManager(graph)
        self.game_state: GameState = GameState()
        self.current_player_index: int = 0
        self.is_final_round: bool = False
        self.is_game_over: bool = False
        
        players_colors: List[Color] = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.BLACK]
        
        for i in range(number_of_players):
            player: Player = Player(players_colors[i])
            self.players.append(player)
        
    def _generate_destination_tickets(self, path: str) -> List[DestinationTicket]:
        df = pd.read_csv(path)
        tickets: List[DestinationTicket] = []
        for _, row in df.iterrows():
            ticket = DestinationTicket(
                city1=row['From'],
                city2=row['To'],
                points=row['Points']
            )
            tickets.append(ticket)
        return tickets
    
    def setup(self):
        self.game_state.setup_deck()

        for player in self.players:
            for _ in range(4):
                card: Optional[Color] = self.game_state.draw_deck_card()
                if card:
                    player.hand.add_card(card)     
        
        self.game_state.setup_destination_tickets(self._generate_destination_tickets("map/tickets.csv"))
        
        for player in self.players:
            tickets: List[DestinationTicket] = self.game_state.draw_destination_tickets(3)
            player.destination_tickets.extend(tickets)
            
    def current_player(self) -> Player:
        return self.players[self.current_player_index]
    
    def next_turn(self) -> None:
        if self.is_game_over:
            return
        
        self.current_player_index = (self.current_player_index + 1) % self.number_of_players
        
        if self.is_final_round and self.current_player_index == 0:
            self.is_game_over = True

    def draw_card_action(self, face_up_card_index: Optional[int]) -> bool:
        player = self.current_player()
        cards_drawn: int = 0

        for _ in range(2):
            if face_up_card_index is not None and face_up_card_index >= 0:
                card = self.game_state.draw_face_up_card(face_up_card_index)
                if card == Color.JOLLY and cards_drawn == 0:
                    cards_drawn = 2
                face_up_card_index = None  # Reset to draw from deck next
            else:
                card = self.game_state.draw_deck_card()
            
            if card:
                player.hand.add_card(card)
                cards_drawn += 1
            
            if cards_drawn >= 2:
                break
        
        self.next_turn()
        return True
    
    def claim_route_action(self, city1: str, city2: str, 
                          color: Color, color_to_use: Color) -> bool:
        player = self.current_player()
        route = self.route_manager.get_route(city1, city2, color)
        
        if not route:
            return False
        
        if player.claim_route(route, color_to_use):
            if player.trains_left <= 2 and not self.is_final_round:
                self.is_final_round = True
            self.next_turn()
            return True
        
        return False
    
    def draw_destination_ticket_action(self, keep_tickets: List[int]) -> bool:
        player = self.current_player()
        drawn = self.game_state.draw_destination_tickets(3)
        
        if len(keep_tickets) < 1:
            return False
        
        kept = [drawn[i] for i in keep_tickets if i < len(drawn)]
        returned = [drawn[i] for i in range(len(drawn)) if i not in keep_tickets]
        
        player.destination_tickets.extend(kept)
        self.game_state.return_destination_tickets(returned)
        
        self.next_turn()
        return True
    
    def calculate_winner(self) -> Tuple[Player, Dict[Color, int]]:
        final_scores: Dict[Color, int] = {}
        player: Color | None = None
        
        
        final_scores = ScoreCalculator.calculate_final_scores(self.players)
        player = ScoreCalculator.longest_route_bonus(self.players)

        if player:
            final_scores[player] += 10

        winner = max(self.players, key=lambda p: final_scores[p.color])
        return winner, final_scores

def main() -> None:
    from map import TicketToRideMap
    
    ttr_map = TicketToRideMap()
    ttr_map.load_graph("map/city_locations.json", "map/routes.csv")
    graph = ttr_map.get_graph()
    
    game = Game(number_of_players=3, graph=graph)
    game.setup()
    
    print("=== TICKET TO RIDE - SIMULAZIONE ===\n")
    
    turn_count = 0
    max_turns = 50
    
    while not game.is_game_over and turn_count < max_turns:
        player = game.current_player()
        print(f"Turno {turn_count + 1} - Giocatore ({player.color.value})")
        print(f"  Carte in mano: {sum(player.hand.cards.values())}")
        print(f"  Vagoni rimasti: {player.trains_left}")
        print(f"  Punteggio attuale: {player.score}")
        
        action_choice = random.choice([0, 1, 2])
        
        if action_choice == 0:
            draw_choice = random.choice([None, 0, 1, 2, 3, 4])
            game.draw_card_action(face_up_card_index=draw_choice)
            print("  Azione: Pesca carte")
        
        elif action_choice == 1:
            available_routes = game.route_manager.get_available_routes(game.number_of_players)

            claimable = []
            for route in available_routes:
                if route.color == Color.GREY:
                    for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]:
                        if player.hand.can_claim_route(route, color):
                            claimable.append((route, color))
                            break
                else:
                    if player.hand.can_claim_route(route, route.color):
                        claimable.append((route, route.color))
            
            if claimable:
                route, color_to_use = random.choice(claimable)
                success = game.claim_route_action(route.city1, route.city2, route.color, color_to_use)
                if success:
                    print(f"  Azione: Reclama rotta {route.city1}-{route.city2} (lunghezza {route.length})")
                else:
                    game.draw_card_action(None)
                    print("  Azione: Pesca carte (fallback)")
            else:
                game.draw_card_action(None)
                print("  Azione: Pesca carte (nessuna rotta disponibile)")
        
        else:
            game.draw_destination_ticket_action([0])
            print("  Azione: Pesca biglietti destinazione")
        
        print()
        turn_count += 1
    
    print("\n=== FINE PARTITA ===\n")
    
    winner, final_scores = game.calculate_winner()
    
    for player in game.players:
        print(f"Giocatore ({player.color.value}):")
        print(f"  Punteggio finale: {final_scores[player.color]}")
        print(f"  Rotte reclamate: {len(player.routes)}")
        print(f"  Biglietti destinazione: {len(player.destination_tickets)}")
        print(f"  Percorso più lungo: {ScoreCalculator.longest_route_bonus([player]) == player.color}")
        print()
    
    print(f"VINCITORE: Giocatore ({winner.color.value})")
    print(f"Punteggio: {final_scores[winner.color]}")


if __name__ == "__main__":
    main()
    