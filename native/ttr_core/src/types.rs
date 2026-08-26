//! Fundamental enums and constants for Ticket to Ride core engine.

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardColor {
    Purple = 0,
    White = 1,
    Blue = 2,
    Yellow = 3,
    Orange = 4,
    Black = 5,
    Red = 6,
    Green = 7,
    Locomotive = 8,
}

impl CardColor {
    pub const ALL_COLORS: [CardColor; 9] = [
        CardColor::Purple,
        CardColor::White,
        CardColor::Blue,
        CardColor::Yellow,
        CardColor::Orange,
        CardColor::Black,
        CardColor::Red,
        CardColor::Green,
        CardColor::Locomotive,
    ];

    pub const STANDARD_COLORS: [CardColor; 8] = [
        CardColor::Purple,
        CardColor::White,
        CardColor::Blue,
        CardColor::Yellow,
        CardColor::Orange,
        CardColor::Black,
        CardColor::Red,
        CardColor::Green,
    ];

    #[inline(always)]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(CardColor::Purple),
            1 => Some(CardColor::White),
            2 => Some(CardColor::Blue),
            3 => Some(CardColor::Yellow),
            4 => Some(CardColor::Orange),
            5 => Some(CardColor::Black),
            6 => Some(CardColor::Red),
            7 => Some(CardColor::Green),
            8 => Some(CardColor::Locomotive),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn as_str(&self) -> &'static str {
        match self {
            CardColor::Purple => "purple",
            CardColor::White => "white",
            CardColor::Blue => "blue",
            CardColor::Yellow => "yellow",
            CardColor::Orange => "orange",
            CardColor::Black => "black",
            CardColor::Red => "red",
            CardColor::Green => "green",
            CardColor::Locomotive => "locomotive",
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TurnState {
    ChoosingInitialTickets = 0,
    Normal = 1,
    DrawingSecondCard = 2,
    ChoosingTickets = 3,
}

impl TurnState {
    pub fn as_str(&self) -> &'static str {
        match self {
            TurnState::ChoosingInitialTickets => "choosing_initial_tickets",
            TurnState::Normal => "normal",
            TurnState::DrawingSecondCard => "drawing_second_card",
            TurnState::ChoosingTickets => "choosing_tickets",
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    DrawVisibleCard = 0,
    DrawHiddenCard = 1,
    ClaimRoute = 2,
    DrawTickets = 3,
    KeepTickets = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Action {
    pub action_type: ActionType,
    pub card_index: u8,         // 0..4 for visible cards, 255 if none
    pub route_id: u8,           // 0..99 for USA routes, 255 if none
    pub color_chosen: u8,       // 0..8 (CardColor as u8), 255 if none
    pub locomotives_count: u8,  // Number of locomotives used
    pub ticket_mask: u32,       // 30-bit mask of kept tickets (u32)
}

impl Action {
    pub fn draw_hidden() -> Self {
        Action {
            action_type: ActionType::DrawHiddenCard,
            card_index: 255,
            route_id: 255,
            color_chosen: 255,
            locomotives_count: 0,
            ticket_mask: 0,
        }
    }

    pub fn draw_visible(slot: u8) -> Self {
        Action {
            action_type: ActionType::DrawVisibleCard,
            card_index: slot,
            route_id: 255,
            color_chosen: 255,
            locomotives_count: 0,
            ticket_mask: 0,
        }
    }

    pub fn claim_route(route_id: u8, color_chosen: u8, locomotives_count: u8) -> Self {
        Action {
            action_type: ActionType::ClaimRoute,
            card_index: 255,
            route_id,
            color_chosen,
            locomotives_count,
            ticket_mask: 0,
        }
    }

    pub fn draw_tickets() -> Self {
        Action {
            action_type: ActionType::DrawTickets,
            card_index: 255,
            route_id: 255,
            color_chosen: 255,
            locomotives_count: 0,
            ticket_mask: 0,
        }
    }

    pub fn keep_tickets(mask: u32) -> Self {
        Action {
            action_type: ActionType::KeepTickets,
            card_index: 255,
            route_id: 255,
            color_chosen: 255,
            locomotives_count: 0,
            ticket_mask: mask,
        }
    }
}
