use ttr_core::board::{get_usa_routes, get_usa_tickets};
use ttr_core::game::GameEngine;
use ttr_core::rules::points_for_length;
use ttr_core::types::{Action, ActionType, TurnState};

#[test]
fn test_game_initialization() {
    let engine = GameEngine::new(42);
    assert_eq!(engine.state.turn_number, 1);
    assert_eq!(engine.state.current_player, 0);
    assert_eq!(engine.state.turn_state, TurnState::ChoosingInitialTickets);
    assert_eq!(engine.state.trains_remaining[0], 45);
    assert_eq!(engine.state.trains_remaining[1], 45);
    assert_eq!(engine.state.total_cards(0), 4);
    assert_eq!(engine.state.total_cards(1), 4);
}

#[test]
fn test_route_scoring_table() {
    assert_eq!(points_for_length(1), 1);
    assert_eq!(points_for_length(2), 2);
    assert_eq!(points_for_length(3), 4);
    assert_eq!(points_for_length(4), 7);
    assert_eq!(points_for_length(5), 10);
    assert_eq!(points_for_length(6), 15);
}

#[test]
fn test_deterministic_simulation_loop() {
    let mut engine1 = GameEngine::new(12345);
    let mut engine2 = GameEngine::new(12345);

    for _ in 0..100 {
        if engine1.state.is_game_over {
            break;
        }
        let acts1 = engine1.valid_actions();
        let acts2 = engine2.valid_actions();
        assert_eq!(acts1.len(), acts2.len());
        assert_eq!(acts1[0], acts2[0]);

        engine1.step(acts1[0]);
        engine2.step(acts2[0]);

        assert_eq!(engine1.state.scores, engine2.state.scores);
        assert_eq!(engine1.state.current_player, engine2.state.current_player);
    }
}
