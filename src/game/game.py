"""Game engine orchestration class for Ticket to Ride."""

from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.card import CardColor, TrainCard, create_standard_train_deck
from src.game.graph import check_ticket_completed, compute_longest_continuous_path
from src.game.maps import load_usa_board
from src.game.player import Player
from src.game.random import SeededRNG
from src.game.rules import GameRules
from src.game.state import GameState, TurnState
from src.game.ticket import DestinationTicket


class Game:
    """Core game engine representing Ticket to Ride.

    100% deterministic, zero ML/web dependencies.
    """

    def __init__(
        self,
        board: Board | None = None,
        tickets_deck: list[DestinationTicket] | None = None,
        num_players: int = 2,
        seed: int = 42,
    ) -> None:
        if board is None:
            self.board, self.initial_tickets = load_usa_board()
        else:
            self.board = board
            self.initial_tickets = tickets_deck or []

        self.num_players = num_players
        self.rng = SeededRNG(seed)
        self.rules = GameRules()
        self.state = GameState(num_players=num_players)

    def reset(self, seed: int | None = None) -> GameState:
        """Reset the game state deterministically."""
        if seed is not None:
            self.rng.seed(seed)

        # Reset board claimed routes
        for r in self.board.routes:
            r.claimed_by = None

        # Create fresh shuffled decks
        train_deck = create_standard_train_deck()
        self.rng.shuffle(train_deck)

        ticket_deck = list(self.initial_tickets)
        self.rng.shuffle(ticket_deck)

        discard_pile: list[TrainCard] = []

        # Initialize players
        players: list[Player] = []
        for i in range(self.num_players):
            p = Player(
                id=f"player_{i}",
                name=f"Player {i+1}",
                trains_remaining=self.rules.INITIAL_TRAINS_PER_PLAYER,
                score=0,
            )
            # Deal 4 train cards
            for _ in range(self.rules.INITIAL_CARDS_PER_PLAYER):
                if train_deck:
                    p.add_card(train_deck.pop())

            # Deal 3 initial destination tickets into pending_tickets
            for _ in range(self.rules.INITIAL_TICKETS_DRAW_COUNT):
                if ticket_deck:
                    p.pending_tickets.append(ticket_deck.pop())

            players.append(p)

        # Deal 5 visible train cards
        visible_cards: list[TrainCard] = []
        for _ in range(self.rules.VISIBLE_CARDS_COUNT):
            if train_deck:
                visible_cards.append(train_deck.pop())

        # Check 3-locomotive flush rule
        visible_cards, train_deck, discard_pile = self._check_and_flush_visible(
            visible_cards, train_deck, discard_pile
        )

        self.state = GameState(
            players=players,
            current_player_index=0,
            turn_state=TurnState.CHOOSING_INITIAL_TICKETS,
            visible_cards=visible_cards,
            train_deck=train_deck,
            discard_pile=discard_pile,
            ticket_deck=ticket_deck,
            turn_number=1,
            is_last_round=False,
            final_turn_player_id=None,
            is_game_over=False,
            winner_id=None,
            num_players=self.num_players,
        )
        return self.state

    def valid_actions(self) -> list[Action]:
        """Compute the list of strictly legal actions for the current player."""
        if self.state.is_game_over:
            return []
        current_p = self.state.current_player
        if not current_p:
            return []
        return self.rules.get_valid_actions(
            player=current_p,
            state=self.state,
            board=self.board,
            num_players=self.num_players,
        )

    def step(self, action: Action) -> GameState:
        """Execute a validated legal action, update state, and advance turns."""
        if self.state.is_game_over:
            return self.state

        current_p = self.state.current_player
        if not current_p:
            return self.state

        if action.action_type == ActionType.KEEP_TICKETS:
            self._handle_keep_tickets(current_p, action)

        elif action.action_type == ActionType.DRAW_HIDDEN_CARD:
            self._handle_draw_hidden_card(current_p)

        elif action.action_type == ActionType.DRAW_VISIBLE_CARD:
            self._handle_draw_visible_card(current_p, action.card_index or 0)

        elif action.action_type == ActionType.CLAIM_ROUTE:
            self._handle_claim_route(current_p, action)

        elif action.action_type == ActionType.DRAW_TICKETS:
            self._handle_draw_tickets(current_p)

        return self.state

    def _handle_keep_tickets(self, player: Player, action: Action) -> None:
        kept_ids = set(action.ticket_ids or ())
        unkept: list[DestinationTicket] = []

        for t in player.pending_tickets:
            if t.id in kept_ids:
                player.tickets.append(t)
            else:
                unkept.append(t)

        player.pending_tickets.clear()
        # Return unkept tickets to the bottom of the ticket deck
        self.state.ticket_deck = unkept + self.state.ticket_deck

        if self.state.turn_state == TurnState.CHOOSING_INITIAL_TICKETS:
            # Advance to next player's initial ticket choice
            next_idx = self.state.current_player_index + 1
            if next_idx >= self.num_players:
                # All players finished initial ticket selection
                self.state.current_player_index = 0
                self.state.turn_state = TurnState.NORMAL
            else:
                self.state.current_player_index = next_idx
        else:
            # Mid-game ticket choice completed
            self.state.turn_state = TurnState.NORMAL
            self._advance_turn()

    def _handle_draw_hidden_card(self, player: Player) -> None:
        card = self._draw_from_train_deck()
        if card:
            player.add_card(card)

        if self.state.turn_state == TurnState.NORMAL:
            self.state.turn_state = TurnState.DRAWING_SECOND_CARD
        elif self.state.turn_state == TurnState.DRAWING_SECOND_CARD:
            self.state.turn_state = TurnState.NORMAL
            self._advance_turn()

    def _handle_draw_visible_card(self, player: Player, card_index: int) -> None:
        if card_index < 0 or card_index >= len(self.state.visible_cards):
            return

        card = self.state.visible_cards.pop(card_index)
        player.add_card(card)

        # Replenish visible slot
        new_card = self._draw_from_train_deck()
        if new_card:
            self.state.visible_cards.insert(card_index, new_card)

        # Check 3-locomotive flush rule
        visible_cards, train_deck, discard_pile = self._check_and_flush_visible(
            self.state.visible_cards,
            self.state.train_deck,
            self.state.discard_pile,
        )
        self.state.visible_cards = visible_cards
        self.state.train_deck = train_deck
        self.state.discard_pile = discard_pile

        if card.is_locomotive() and self.state.turn_state == TurnState.NORMAL:
            # Drawing a locomotive as the first card immediately consumes the full turn
            self.state.turn_state = TurnState.NORMAL
            self._advance_turn()
        elif self.state.turn_state == TurnState.NORMAL:
            self.state.turn_state = TurnState.DRAWING_SECOND_CARD
        elif self.state.turn_state == TurnState.DRAWING_SECOND_CARD:
            self.state.turn_state = TurnState.NORMAL
            self._advance_turn()

    def _handle_claim_route(self, player: Player, action: Action) -> None:
        route = self.board.get_route(action.route_id or "")
        if not route:
            return

        locos = action.locomotives_count
        color_cards = route.length - locos
        spend_dict: dict[CardColor, int] = {}

        if action.color_chosen and color_cards > 0:
            spend_dict[action.color_chosen] = color_cards
        if locos > 0:
            spend_dict[CardColor.LOCOMOTIVE] = locos

        player.remove_cards(spend_dict)

        # Move spent cards to discard pile
        for c, count in spend_dict.items():
            self.state.discard_pile.extend([TrainCard(color=c) for _ in range(count)])

        # Place trains & update route
        player.trains_remaining -= route.length
        route.claimed_by = player.id
        player.claimed_route_ids.append(route.id)

        # Award immediate route score
        player.score += self.rules.points_for_route_length(route.length)

        # Check end game trigger (2 or fewer trains remaining)
        if (
            player.trains_remaining <= 2
            and not self.state.is_last_round
        ):
            self.state.is_last_round = True
            self.state.final_turn_player_id = player.id

        self.state.turn_state = TurnState.NORMAL
        self._advance_turn()

    def _handle_draw_tickets(self, player: Player) -> None:
        draw_count = min(self.rules.MIDGAME_TICKETS_DRAW_COUNT, len(self.state.ticket_deck))
        for _ in range(draw_count):
            if self.state.ticket_deck:
                player.pending_tickets.append(self.state.ticket_deck.pop())

        self.state.turn_state = TurnState.CHOOSING_TICKETS

    def _advance_turn(self) -> None:
        """Advance turn to the next player and check for game over."""
        current_idx = self.state.current_player_index
        next_idx = (current_idx + 1) % self.num_players

        # If last round is active and the next player is the one who triggered final round: GAME OVER
        if self.state.is_last_round and self.state.players[next_idx].id == self.state.final_turn_player_id:
            self._end_game()
            return

        self.state.current_player_index = next_idx
        if next_idx == 0:
            self.state.turn_number += 1

    def _end_game(self) -> None:
        """Compute final scores, tickets, longest path bonus, and winner."""
        self.state.is_game_over = True

        # Calculate ticket completions and longest path
        longest_paths: dict[str, int] = {}
        for p in self.state.players:
            p_routes = [self.board.get_route(r_id) for r_id in p.claimed_route_ids]
            valid_p_routes = [r for r in p_routes if r is not None]

            # Score destination tickets
            for ticket in p.tickets:
                if check_ticket_completed(valid_p_routes, ticket):
                    p.score += ticket.points
                else:
                    p.score -= ticket.points

            # Longest path
            longest_paths[p.id] = compute_longest_continuous_path(valid_p_routes)

        # Award 10-point bonus to player(s) with maximum longest path
        if longest_paths:
            max_len = max(longest_paths.values())
            if max_len > 0:
                for p in self.state.players:
                    if longest_paths[p.id] == max_len:
                        p.score += self.rules.LONGEST_PATH_BONUS_POINTS

        # Determine winner
        best_player = max(self.state.players, key=lambda p: p.score)
        self.state.winner_id = best_player.id

    def _draw_from_train_deck(self) -> TrainCard | None:
        if not self.state.train_deck and self.state.discard_pile:
            self.state.train_deck = list(self.state.discard_pile)
            self.state.discard_pile.clear()
            self.rng.shuffle(self.state.train_deck)

        if self.state.train_deck:
            return self.state.train_deck.pop()
        return None

    def _check_and_flush_visible(
        self,
        visible_cards: list[TrainCard],
        train_deck: list[TrainCard],
        discard_pile: list[TrainCard],
    ) -> tuple[list[TrainCard], list[TrainCard], list[TrainCard]]:
        """Flush visible cards if 3 or more locomotives are face up."""
        while self.rules.should_flush_visible_cards(visible_cards):
            discard_pile.extend(visible_cards)
            visible_cards.clear()

            # Deal 5 new cards
            for _ in range(self.rules.VISIBLE_CARDS_COUNT):
                if not train_deck and discard_pile:
                    train_deck = list(discard_pile)
                    discard_pile.clear()
                    self.rng.shuffle(train_deck)

                if train_deck:
                    visible_cards.append(train_deck.pop())

        return visible_cards, train_deck, discard_pile
