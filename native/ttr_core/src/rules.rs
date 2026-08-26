//! Official game rules, score values, validation and action generation.

use crate::board::{RouteDef, USA_NUM_ROUTES, USA_NUM_TICKETS};
use crate::state::GameStateCompact;
use crate::types::{Action, TurnState};

pub const ROUTE_POINTS: [u8; 7] = [0, 1, 2, 4, 7, 10, 15]; // length -> points

#[inline(always)]
pub fn points_for_length(len: u8) -> u8 {
    if (len as usize) < ROUTE_POINTS.len() {
        ROUTE_POINTS[len as usize]
    } else {
        0
    }
}

pub fn get_valid_actions(state: &GameStateCompact, routes: &[RouteDef; USA_NUM_ROUTES]) -> Vec<Action> {
    let mut actions = Vec::with_capacity(128);
    let p = state.current_player as usize;

    match state.turn_state {
        TurnState::ChoosingInitialTickets => {
            let pending_mask = state.tickets_pending[p];
            let mut pending_indices = [0u8; 3];
            let mut count = 0;
            for i in 0..30 {
                if (pending_mask & (1 << i)) != 0 {
                    pending_indices[count] = i as u8;
                    count += 1;
                    if count == 3 {
                        break;
                    }
                }
            }

            if count >= 3 {
                // Keep 2
                actions.push(Action::keep_tickets((1u32 << pending_indices[0]) | (1u32 << pending_indices[1])));
                actions.push(Action::keep_tickets((1u32 << pending_indices[0]) | (1u32 << pending_indices[2])));
                actions.push(Action::keep_tickets((1u32 << pending_indices[1]) | (1u32 << pending_indices[2])));
                // Keep all 3
                actions.push(Action::keep_tickets(
                    (1u32 << pending_indices[0]) | (1u32 << pending_indices[1]) | (1u32 << pending_indices[2]),
                ));
            } else if count == 2 {
                actions.push(Action::keep_tickets((1u32 << pending_indices[0]) | (1u32 << pending_indices[1])));
            } else if count > 0 {
                actions.push(Action::keep_tickets(pending_mask));
            }
        }
        TurnState::ChoosingTickets => {
            let pending_mask = state.tickets_pending[p];
            let mut pending_indices = [0u8; 3];
            let mut count = 0;
            for i in 0..30 {
                if (pending_mask & (1 << i)) != 0 {
                    pending_indices[count] = i as u8;
                    count += 1;
                    if count == 3 {
                        break;
                    }
                }
            }

            if count == 3 {
                actions.push(Action::keep_tickets(1u32 << pending_indices[0]));
                actions.push(Action::keep_tickets(1u32 << pending_indices[1]));
                actions.push(Action::keep_tickets(1u32 << pending_indices[2]));
                actions.push(Action::keep_tickets((1u32 << pending_indices[0]) | (1u32 << pending_indices[1])));
                actions.push(Action::keep_tickets((1u32 << pending_indices[0]) | (1u32 << pending_indices[2])));
                actions.push(Action::keep_tickets((1u32 << pending_indices[1]) | (1u32 << pending_indices[2])));
                actions.push(Action::keep_tickets(
                    (1u32 << pending_indices[0]) | (1u32 << pending_indices[1]) | (1u32 << pending_indices[2]),
                ));
            } else {
                for mask in 1..=(1 << count) - 1 {
                    let mut actual_mask = 0u32;
                    for bit in 0..count {
                        if (mask & (1 << bit)) != 0 {
                            actual_mask |= 1u32 << pending_indices[bit];
                        }
                    }
                    actions.push(Action::keep_tickets(actual_mask));
                }
            }
        }
        TurnState::DrawingSecondCard => {
            for slot in 0..5 {
                let card = state.visible_cards[slot];
                if card != 255 && card != 8 {
                    actions.push(Action::draw_visible(slot as u8));
                }
            }
            if state.deck_ptr < state.deck.len() as u8 || state.discard_counts.iter().sum::<u8>() > 0 {
                actions.push(Action::draw_hidden());
            }
        }
        TurnState::Normal => {
            if state.deck_ptr < state.deck.len() as u8 || state.discard_counts.iter().sum::<u8>() > 0 {
                actions.push(Action::draw_hidden());
            }

            for slot in 0..5 {
                let card = state.visible_cards[slot];
                if card != 255 {
                    actions.push(Action::draw_visible(slot as u8));
                }
            }

            let trains_left = state.trains_remaining[p];
            let hand = &state.hands[p];
            let locos = hand[8];

            for r in routes.iter() {
                if state.is_route_claimed(r.id) || trains_left < r.length {
                    continue;
                }
                if r.double_pair_id != 255 && state.is_route_claimed(r.double_pair_id) {
                    continue;
                }

                if r.color != 255 {
                    let col = r.color as usize;
                    let have_col = hand[col];
                    if have_col + locos >= r.length {
                        let needed_locos = if have_col >= r.length { 0 } else { r.length - have_col };
                        actions.push(Action::claim_route(r.id, col as u8, needed_locos));
                    }
                } else {
                    for col in 0..8 {
                        let have_col = hand[col];
                        if have_col > 0 && have_col + locos >= r.length {
                            let needed_locos = if have_col >= r.length { 0 } else { r.length - have_col };
                            actions.push(Action::claim_route(r.id, col as u8, needed_locos));
                        }
                    }
                    if locos >= r.length {
                        actions.push(Action::claim_route(r.id, 8, r.length));
                    }
                }
            }

            if state.ticket_deck_ptr < USA_NUM_TICKETS as u8 {
                actions.push(Action::draw_tickets());
            }
        }
    }

    if actions.is_empty() {
        actions.push(Action::draw_hidden());
    }

    actions
}
