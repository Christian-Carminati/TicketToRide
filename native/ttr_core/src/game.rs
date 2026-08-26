//! High-performance deterministic Game engine.

use crate::board::{get_usa_routes, get_usa_tickets, RouteDef, TicketDef, USA_NUM_CITIES, USA_NUM_ROUTES, USA_NUM_TICKETS};
use crate::rules::{get_valid_actions, points_for_length};
use crate::state::{GameStateCompact, DECK_SIZE};
use crate::types::{Action, ActionType, TurnState};

// Fast xoshiro256+ RNG
#[inline(always)]
fn next_rand(state: &mut [u64; 2]) -> u64 {
    let s0 = state[0];
    let mut s1 = state[1];
    let result = s0.wrapping_add(s1);
    s1 ^= s0;
    state[0] = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
    state[1] = s1.rotate_left(37);
    result
}

#[derive(Clone)]
pub struct GameEngine {
    pub state: GameStateCompact,
    pub routes: [RouteDef; USA_NUM_ROUTES],
    pub tickets: [TicketDef; USA_NUM_TICKETS],
}

impl GameEngine {
    pub fn new(seed: u64) -> Self {
        let mut engine = GameEngine {
            state: GameStateCompact::new(),
            routes: get_usa_routes(),
            tickets: get_usa_tickets(),
        };
        engine.reset(seed);
        engine
    }

    pub fn reset(&mut self, seed: u64) {
        self.state = GameStateCompact::new();
        self.state.rng_state = [seed ^ 0x9E3779B97F4A7C15, seed.rotate_left(32) ^ 0x6A09E667F3BCC908];

        // 1. Create standard train deck (96 standard: 12 of each 8 colors, + 14 locomotives)
        let mut deck = [0u8; DECK_SIZE];
        let mut idx = 0;
        for c in 0..8 {
            for _ in 0..12 {
                deck[idx] = c;
                idx += 1;
            }
        }
        for _ in 0..14 {
            deck[idx] = 8;
            idx += 1;
        }

        // Shuffle train deck with deterministic Fisher-Yates
        for i in (1..DECK_SIZE).rev() {
            let j = (next_rand(&mut self.state.rng_state) % ((i + 1) as u64)) as usize;
            deck.swap(i, j);
        }
        self.state.deck = deck;
        self.state.deck_ptr = 0;

        // 2. Shuffle destination tickets deck (30 tickets)
        let mut t_deck = [0u8; USA_NUM_TICKETS];
        for i in 0..USA_NUM_TICKETS {
            t_deck[i] = i as u8;
        }
        for i in (1..USA_NUM_TICKETS).rev() {
            let j = (next_rand(&mut self.state.rng_state) % ((i + 1) as u64)) as usize;
            t_deck.swap(i, j);
        }
        self.state.ticket_deck = t_deck;
        self.state.ticket_deck_ptr = 0;

        // 3. Deal 4 cards to each of the 2 players
        for p in 0..2 {
            for _ in 0..4 {
                let card = self.draw_train_card_internal();
                self.state.hands[p][card as usize] += 1;
            }
            // Deal 3 initial pending tickets
            for _ in 0..3 {
                if self.state.ticket_deck_ptr < USA_NUM_TICKETS as u8 {
                    let tid = self.state.ticket_deck[self.state.ticket_deck_ptr as usize];
                    self.state.ticket_deck_ptr += 1;
                    self.state.tickets_pending[p] |= 1 << tid;
                }
            }
        }

        // 4. Deal 5 visible cards
        for slot in 0..5 {
            self.state.visible_cards[slot] = self.draw_train_card_internal();
        }
        self.check_and_flush_visible();
    }

    #[inline(always)]
    fn draw_train_card_internal(&mut self) -> u8 {
        if self.state.deck_ptr >= DECK_SIZE as u8 {
            self.reshuffle_discard();
        }
        if self.state.deck_ptr < DECK_SIZE as u8 {
            let card = self.state.deck[self.state.deck_ptr as usize];
            self.state.deck_ptr += 1;
            card
        } else {
            // Deck empty fallback
            8
        }
    }

    fn reshuffle_discard(&mut self) {
        let total_discard: u8 = self.state.discard_counts.iter().sum();
        if total_discard == 0 {
            return;
        }
        let mut idx = 0;
        for c in 0..9 {
            for _ in 0..self.state.discard_counts[c] {
                if idx < DECK_SIZE {
                    self.state.deck[idx] = c as u8;
                    idx += 1;
                }
            }
            self.state.discard_counts[c] = 0;
        }
        // Shuffle new deck
        for i in (1..idx).rev() {
            let j = (next_rand(&mut self.state.rng_state) % ((i + 1) as u64)) as usize;
            self.state.deck.swap(i, j);
        }
        self.state.deck_ptr = 0;
    }

    fn check_and_flush_visible(&mut self) {
        let mut flushes = 0;
        while flushes < 10 {
            let loco_count = self.state.visible_cards.iter().filter(|&&c| c == 8).count();
            if loco_count < 3 {
                break;
            }
            flushes += 1;
            for slot in 0..5 {
                let card = self.state.visible_cards[slot];
                if card != 255 {
                    self.state.discard_counts[card as usize] += 1;
                    self.state.visible_cards[slot] = 255;
                }
            }
            for slot in 0..5 {
                self.state.visible_cards[slot] = self.draw_train_card_internal();
            }
        }
    }

    pub fn valid_actions(&self) -> Vec<Action> {
        if self.state.is_game_over {
            return Vec::new();
        }
        get_valid_actions(&self.state, &self.routes)
    }

    pub fn step(&mut self, action: Action) {
        if self.state.is_game_over {
            return;
        }

        let p = self.state.current_player as usize;

        match action.action_type {
            ActionType::KeepTickets => {
                // Add kept tickets from pending to tickets_held
                self.state.tickets_held[p] |= action.ticket_mask as u32;
                self.state.tickets_pending[p] = 0;

                if self.state.turn_state == TurnState::ChoosingInitialTickets {
                    if p == 0 {
                        self.state.current_player = 1;
                    } else {
                        self.state.current_player = 0;
                        self.state.turn_state = TurnState::Normal;
                    }
                } else {
                    self.advance_turn();
                }
            }
            ActionType::DrawTickets => {
                // Deal 3 tickets to pending
                for _ in 0..3 {
                    if self.state.ticket_deck_ptr < USA_NUM_TICKETS as u8 {
                        let tid = self.state.ticket_deck[self.state.ticket_deck_ptr as usize];
                        self.state.ticket_deck_ptr += 1;
                        self.state.tickets_pending[p] |= 1 << tid;
                    }
                }
                self.state.turn_state = TurnState::ChoosingTickets;
            }
            ActionType::DrawHiddenCard => {
                let card = self.draw_train_card_internal();
                self.state.hands[p][card as usize] += 1;

                if self.state.turn_state == TurnState::Normal {
                    self.state.turn_state = TurnState::DrawingSecondCard;
                } else {
                    self.advance_turn();
                }
            }
            ActionType::DrawVisibleCard => {
                let slot = action.card_index as usize;
                let card = self.state.visible_cards[slot];
                self.state.hands[p][card as usize] += 1;

                // Replace drawn slot from train deck
                self.state.visible_cards[slot] = self.draw_train_card_internal();
                self.check_and_flush_visible();

                if card == 8 && self.state.turn_state == TurnState::Normal {
                    // Drawing a locomotive in Normal state consumes the whole turn
                    self.advance_turn();
                } else if self.state.turn_state == TurnState::Normal {
                    self.state.turn_state = TurnState::DrawingSecondCard;
                } else {
                    self.advance_turn();
                }
            }
            ActionType::ClaimRoute => {
                let r_id = action.route_id as usize;
                let route = self.routes[r_id];
                let col = action.color_chosen as usize;
                let locos = action.locomotives_count;
                let regular_cards = route.length - locos;

                if col < 8 && regular_cards > 0 {
                    self.state.hands[p][col] -= regular_cards;
                    self.state.discard_counts[col] += regular_cards;
                }
                if locos > 0 {
                    self.state.hands[p][8] -= locos;
                    self.state.discard_counts[8] += locos;
                }

                self.state.trains_remaining[p] -= route.length;
                self.state.scores[p] += points_for_length(route.length) as i16;
                self.state.claimed_routes[p] |= 1u128 << r_id;

                // Incremental DSU update O(1)
                self.state.dsu[p].union(route.city_a as usize, route.city_b as usize);

                // Check last round trigger (trains <= 2)
                if self.state.trains_remaining[p] <= 2 && !self.state.is_last_round {
                    self.state.is_last_round = true;
                    self.state.final_turn_player = p as u8;
                }

                self.advance_turn();
            }
        }
    }

    fn advance_turn(&mut self) {
        if self.state.is_last_round && self.state.current_player == self.state.final_turn_player {
            self.end_game();
            return;
        }

        self.state.current_player = 1 - self.state.current_player;
        self.state.turn_state = TurnState::Normal;
        self.state.turn_number += 1;

        if self.state.turn_number >= 300 {
            self.end_game();
        }
    }

    pub fn end_game(&mut self) {
        self.state.is_game_over = true;

        // 1. Destination Tickets scoring
        for p in 0..2 {
            let held = self.state.tickets_held[p];
            for i in 0..USA_NUM_TICKETS {
                if (held & (1 << i)) != 0 {
                    let t = self.tickets[i];
                    let completed = self.state.dsu[p].is_connected(t.city_a as usize, t.city_b as usize);
                    if completed {
                        self.state.scores[p] += t.points as i16;
                    } else {
                        self.state.scores[p] -= t.points as i16;
                    }
                }
            }
        }

        // 2. Longest Continuous Path Bonus (10 points)
        let len0 = self.compute_longest_path(0);
        let len1 = self.compute_longest_path(1);
        if len0 > len1 {
            self.state.scores[0] += 10;
        } else if len1 > len0 {
            self.state.scores[1] += 10;
        } else if len0 > 0 {
            self.state.scores[0] += 10;
            self.state.scores[1] += 10;
        }

        // 3. Determine winner
        if self.state.scores[0] > self.state.scores[1] {
            self.state.winner_id = 0;
        } else if self.state.scores[1] > self.state.scores[0] {
            self.state.winner_id = 1;
        } else {
            self.state.winner_id = 2; // Draw
        }
    }

    pub fn compute_longest_path(&self, player: usize) -> u8 {
        let mask = self.state.claimed_routes[player];
        if mask == 0 {
            return 0;
        }

        // Build adjacency: city -> [(neighbor_city, route_id, length)]
        let mut adj = vec![Vec::<(u8, u8, u8)>::new(); USA_NUM_CITIES];
        let mut cities = Vec::with_capacity(36);

        for r in self.routes.iter() {
            if (mask & (1u128 << r.id)) != 0 {
                adj[r.city_a as usize].push((r.city_b, r.id, r.length));
                adj[r.city_b as usize].push((r.city_a, r.id, r.length));
                if !cities.contains(&r.city_a) {
                    cities.push(r.city_a);
                }
                if !cities.contains(&r.city_b) {
                    cities.push(r.city_b);
                }
            }
        }

        let mut max_len = 0u8;

        fn dfs(
            curr: u8,
            curr_len: u8,
            visited_edges: u128,
            adj: &[Vec<(u8, u8, u8)>],
            max_len: &mut u8,
        ) {
            if curr_len > *max_len {
                *max_len = curr_len;
            }
            for &(nbr, edge_id, len) in &adj[curr as usize] {
                let e_mask = 1u128 << edge_id;
                if (visited_edges & e_mask) == 0 {
                    dfs(nbr, curr_len + len, visited_edges | e_mask, adj, max_len);
                }
            }
        }

        for &start_city in &cities {
            dfs(start_city, 0, 0, &adj, &mut max_len);
        }

        max_len
    }
}
