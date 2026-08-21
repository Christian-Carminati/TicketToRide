"""Determinization module for Information Set Monte Carlo Tree Search (ISMCTS / SO-MCTS).

This module handles partial observability by sampling plausible world states consistent
with all publicly known information, adhering strictly to POMDP anti-leakage invariants.
"""

from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.random import SeededRNG


def determinize_game(game: Game, root_player_id: str, rng: SeededRNG) -> Game:
    """Sample a determinized game state consistent with public observations.

    Mathematical Concept:
        In imperfect information games, Single-Observer MCTS (SO-MCTS) samples a state
        s ~ P(S | H_pub, h_root) from the belief state conditioned on public history
        and private knowledge of the root player.

    Algorithm:
        1. Deep clone the game.
        2. Pool all cards hidden from root_player (cards in opponent hands + train deck).
        3. Shuffle the hidden card pool uniformly at random using `rng`.
        4. Deal back to each opponent exactly the number of cards they currently hold.
        5. Put the remainder back into the train deck.
        6. Similarly redistribute unknown destination tickets to opponents.

    Args:
        game: Current game state.
        root_player_id: The ID of the observing agent (e.g. "player_0").
        rng: Seeded pseudo-random number generator.

    Returns:
        A cloned and determinized Game instance ready for tree simulation.
    """
    det_game = game.clone(rng_seed=rng.randint(0, 1_000_000_000))
    state = det_game.state

    # 1. Collect all hidden train cards (opponents' cards + train deck)
    hidden_cards: list[TrainCard] = list(state.train_deck)
    for p in state.players:
        if p.id != root_player_id:
            for color, count in p.cards.items():
                hidden_cards.extend([TrainCard(color=color) for _ in range(count)])
            # Reset opponent card inventory
            p.cards = {c: 0 for c in CardColor}

    # 2. Shuffle hidden card pool
    rng.shuffle(hidden_cards)

    # 3. Redistribute cards to opponents
    for p in state.players:
        if p.id != root_player_id:
            orig_p = next(op for op in game.state.players if op.id == p.id)
            target_count = orig_p.total_cards()
            for _ in range(target_count):
                if hidden_cards:
                    p.add_card(hidden_cards.pop())

    # 4. Remaining cards form the train deck
    state.train_deck = hidden_cards

    # 5. Handle opponent tickets if any unknown tickets in deck
    hidden_tickets = list(state.ticket_deck)
    for p in state.players:
        if p.id != root_player_id:
            hidden_tickets.extend(p.tickets)
            p.tickets = []

    rng.shuffle(hidden_tickets)

    for p in state.players:
        if p.id != root_player_id:
            orig_p = next(op for op in game.state.players if op.id == p.id)
            target_ticket_count = len(orig_p.tickets)
            for _ in range(target_ticket_count):
                if hidden_tickets:
                    p.tickets.append(hidden_tickets.pop())

    state.ticket_deck = hidden_tickets

    return det_game
