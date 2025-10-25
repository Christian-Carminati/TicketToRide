from dataclasses import dataclass
import enum
import random
from typing import List, Dict, Optional, Tuple
from map import TicketToRideMap
import networkx as nx

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
    def __init__(self, cards: Dict[Color, int] = {}) -> None:
        self.cards: Dict[Color, int] = cards

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
            
    def remove_card(self, card: Color, count: int = 1) -> None:
        """
        Remove a card from the player's hand.

        :param card: A Color object representing the card to remove.
        :type card: Color
        :param count: The number of cards to remove.
        :type count: int
        :return: None
        """
        if card in self.cards and self.cards[card] >= count:
            self.cards[card] -= count
            if self.cards[card] == 0:
                del self.cards[card]
        else:
            raise ValueError(f"Not enough {card} cards to remove.") 
        
    def can_claim_route(self, route: Route, color_to_use: Color) -> bool:
        """
        Check if the player can claim the given route with their current hand.

        :param route: A Route object representing the route to claim.
        :type route: Route
        :param color_to_use: The color of cards to use to claim the route.
        :type color_to_use: Color
        :return: Whether the player can claim the route.
        :rtype: bool
        """
        if route.owner is not None:
            return False
        
        if route.color == Color.GREY:
            total = self.cards.get(color_to_use, 0) + self.cards.get(Color.JOLLY, 0)
            return total >= route.length
        
        total = self.cards.get(route.color, 0) + self.cards.get(Color.JOLLY, 0)
        return total >= route.length
        
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
        :return: Whether the player can claim the route.
        :rtype: bool
        """
        if not self.hand.can_claim_route(route, color_to_use):
            return False
        
        if route.color == Color.GREY:
            needed_color: Color = color_to_use
        else:
            needed_color: Color = route.color
        
        cards_to_remove: List[Color] = []
        available: int = self.hand.cards.get(needed_color, 0)
        jolly_needed: int = max(0, route.length - available)
        color_needed: int = route.length - jolly_needed
        
        if color_needed > 0:
            cards_to_remove.extend([needed_color] * color_needed)
        if jolly_needed > 0:
            cards_to_remove.extend([Color.JOLLY] * jolly_needed)
        
        for card in cards_to_remove:
            count: int = cards_to_remove.count(card)
            if not self.hand.remove_card(card, count):
                return False
            cards_to_remove = [c for c in cards_to_remove if c != card]
        
        route.owner = self.color
        self.routes.append(route)
        self.trains_left -= route.length
        
        points_table: Dict[int, int] = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
        self.score += points_table.get(route.length, 0)
        
        return True

class GameState:
    def __init__(self) -> None:
        self.face_up_cards: List[Color] = []
        self.deck: Deck = Deck()
        self.discard_pile: list[Color] = []
        self.destination_ticket_deck: List[DestinationTicket] = []

    def setup_deck(self) -> None:
        self.deck = Deck.full_deck()
        self.face_up_cards = []
        for _ in range(5):
            card = self.deck.draw()
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
        return self.deck.draw()
    
    def reshuffle_discard_into_deck(self) -> None:
        """
        Reshuffle the discard pile back into the deck.
        
        :return: None
        """
        random.shuffle(self.discard_pile)
        self.deck.add_card(self.discard_pile)
        self.discard_pile = []
        
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
    
    def get_route(self, city1: str, city2: str) -> Optional[Route]:
        """
        Get a route between two cities.

        :param city1: The first city.
        :type city1: str
        :param city2: The second city.
        :type city2: str
        :return: The Route object if found, else None.
        :rtype: Optional[Route]
        """
        for route in self.routes:
            if (route.city1 == city1 and route.city2 == city2) or (route.city1 == city2 and route.city2 == city1):
                return route
        return None


if __name__ == "__main__":
    pass