//! Compact, cache-aligned GameState for zero-overhead cloning and SIMD simulation.

use crate::board::USA_NUM_TICKETS;
use crate::dsu::DisjointSet;
use crate::types::TurnState;

pub const DECK_SIZE: usize = 110;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameStateCompact {
    pub hands: [[u8; 9]; 2],
    pub trains_remaining: [u8; 2],
    pub scores: [i16; 2],

    pub claimed_routes: [u128; 2],

    pub tickets_held: [u32; 2],
    pub tickets_pending: [u32; 2],

    pub visible_cards: [u8; 5],

    pub deck: [u8; DECK_SIZE],
    pub deck_ptr: u8,
    pub discard_counts: [u8; 9],

    pub ticket_deck: [u8; USA_NUM_TICKETS],
    pub ticket_deck_ptr: u8,

    pub turn_number: u16,
    pub current_player: u8,
    pub turn_state: TurnState,
    pub is_last_round: bool,
    pub final_turn_player: u8,
    pub is_game_over: bool,
    pub winner_id: i8,

    pub rng_state: [u64; 2],

    pub dsu: [DisjointSet; 2],
}

impl GameStateCompact {
    pub fn new() -> Self {
        GameStateCompact {
            hands: [[0; 9]; 2],
            trains_remaining: [45, 45],
            scores: [0, 0],
            claimed_routes: [0, 0],
            tickets_held: [0, 0],
            tickets_pending: [0, 0],
            visible_cards: [255; 5],
            deck: [0; DECK_SIZE],
            deck_ptr: 0,
            discard_counts: [0; 9],
            ticket_deck: [0; USA_NUM_TICKETS],
            ticket_deck_ptr: 0,
            turn_number: 1,
            current_player: 0,
            turn_state: TurnState::ChoosingInitialTickets,
            is_last_round: false,
            final_turn_player: 255,
            is_game_over: false,
            winner_id: -1,
            rng_state: [0x123456789ABCDEF0, 0x0FEDCBA987654321],
            dsu: [DisjointSet::new(), DisjointSet::new()],
        }
    }

    #[inline(always)]
    pub fn total_cards(&self, player: usize) -> u8 {
        self.hands[player].iter().sum()
    }

    #[inline(always)]
    pub fn is_route_claimed(&self, route_id: u8) -> bool {
        let mask = 1u128 << route_id;
        ((self.claimed_routes[0] | self.claimed_routes[1]) & mask) != 0
    }

    #[inline(always)]
    pub fn route_owner(&self, route_id: u8) -> Option<u8> {
        let mask = 1u128 << route_id;
        if (self.claimed_routes[0] & mask) != 0 {
            Some(0)
        } else if (self.claimed_routes[1] & mask) != 0 {
            Some(1)
        } else {
            None
        }
    }
}
