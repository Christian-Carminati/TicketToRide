//! PyO3 bindings exposing high-performance native core to Python & PyTorch.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::board::USA_NUM_TICKETS;
use crate::game::GameEngine;
use crate::types::{Action, ActionType, CardColor};

#[pyclass]
#[derive(Clone)]
pub struct PyGame {
    pub engine: GameEngine,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (seed=42))]
    pub fn new(seed: u64) -> Self {
        PyGame {
            engine: GameEngine::new(seed),
        }
    }

    #[pyo3(signature = (seed=None))]
    pub fn reset(&mut self, seed: Option<u64>) {
        let s = seed.unwrap_or(42);
        self.engine.reset(s);
    }

    pub fn is_game_over(&self) -> bool {
        self.engine.state.is_game_over
    }

    pub fn current_player_index(&self) -> usize {
        self.engine.state.current_player as usize
    }

    pub fn turn_number(&self) -> u16 {
        self.engine.state.turn_number
    }

    pub fn turn_state(&self) -> &'static str {
        self.engine.state.turn_state.as_str()
    }

    pub fn scores(&self) -> (i16, i16) {
        (self.engine.state.scores[0], self.engine.state.scores[1])
    }

    pub fn winner_id(&self) -> i8 {
        self.engine.state.winner_id
    }

    pub fn trains_remaining(&self) -> (u8, u8) {
        (
            self.engine.state.trains_remaining[0],
            self.engine.state.trains_remaining[1],
        )
    }

    pub fn visible_cards(&self) -> Vec<Option<&'static str>> {
        self.engine
            .state
            .visible_cards
            .iter()
            .map(|&c| {
                if c < 9 {
                    Some(CardColor::from_u8(c).unwrap().as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn player_hand(&self, player: usize) -> Vec<u8> {
        self.engine.state.hands[player].to_vec()
    }

    pub fn player_tickets(&self, player: usize) -> Vec<u8> {
        let held = self.engine.state.tickets_held[player];
        let mut t_list = Vec::new();
        for i in 0..USA_NUM_TICKETS {
            if (held & (1 << i)) != 0 {
                t_list.push(i as u8);
            }
        }
        t_list
    }

    pub fn is_ticket_completed(&self, player: usize, ticket_id: usize) -> bool {
        if ticket_id >= USA_NUM_TICKETS {
            return false;
        }
        let t = self.engine.tickets[ticket_id];
        self.engine.state.dsu[player].is_connected(t.city_a as usize, t.city_b as usize)
    }

    pub fn valid_actions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let actions = self.engine.valid_actions();
        let py_list = PyList::empty_bound(py);

        for act in actions {
            let dict = PyDict::new_bound(py);
            match act.action_type {
                ActionType::DrawHiddenCard => {
                    dict.set_item("action_type", "draw_hidden_card")?;
                }
                ActionType::DrawVisibleCard => {
                    dict.set_item("action_type", "draw_visible_card")?;
                    dict.set_item("card_index", act.card_index)?;
                }
                ActionType::ClaimRoute => {
                    dict.set_item("action_type", "claim_route")?;
                    dict.set_item("route_id", format!("route_{}", act.route_id))?;
                    if act.color_chosen < 9 {
                        dict.set_item("color_chosen", CardColor::from_u8(act.color_chosen).unwrap().as_str())?;
                    }
                    dict.set_item("locomotives_count", act.locomotives_count)?;
                }
                ActionType::DrawTickets => {
                    dict.set_item("action_type", "draw_tickets")?;
                }
                ActionType::KeepTickets => {
                    dict.set_item("action_type", "keep_tickets")?;
                    let mut t_ids = Vec::new();
                    for i in 0..30 {
                        if (act.ticket_mask & (1 << i)) != 0 {
                            t_ids.push(format!("ticket_{}", i));
                        }
                    }
                    dict.set_item("ticket_ids", t_ids)?;
                }
            }
            py_list.append(dict)?;
        }
        Ok(py_list.into())
    }

    pub fn step_dict(&mut self, action_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let action_type_str: String = action_dict.get_item("action_type")?.unwrap().extract()?;
        let action = match action_type_str.as_str() {
            "draw_hidden_card" => Action::draw_hidden(),
            "draw_visible_card" => {
                let card_idx: u8 = action_dict.get_item("card_index")?.unwrap().extract()?;
                Action::draw_visible(card_idx)
            }
            "claim_route" => {
                let r_str: String = action_dict.get_item("route_id")?.unwrap().extract()?;
                let r_id: u8 = r_str.strip_prefix("route_").unwrap_or(&r_str).parse().unwrap_or(0);
                let col_str: Option<String> = action_dict.get_item("color_chosen")?.map(|v| v.extract().unwrap());
                let col_u8 = match col_str.as_deref() {
                    Some("purple") => 0,
                    Some("white") => 1,
                    Some("blue") => 2,
                    Some("yellow") => 3,
                    Some("orange") => 4,
                    Some("black") => 5,
                    Some("red") => 6,
                    Some("green") => 7,
                    Some("locomotive") => 8,
                    _ => 255,
                };
                let locos: u8 = action_dict
                    .get_item("locomotives_count")?
                    .map(|v| v.extract().unwrap_or(0))
                    .unwrap_or(0);
                Action::claim_route(r_id, col_u8, locos)
            }
            "draw_tickets" => Action::draw_tickets(),
            "keep_tickets" => {
                let mut mask = 0u32;
                if let Some(t_item) = action_dict.get_item("ticket_ids")? {
                    if let Ok(t_list) = t_item.downcast::<PyList>() {
                        for item in t_list.iter() {
                            let s: String = item.extract()?;
                            let tid: usize = s.strip_prefix("ticket_").unwrap_or(&s).parse().unwrap_or(0);
                            if tid < 30 {
                                mask |= 1 << tid;
                            }
                        }
                    }
                }
                Action::keep_tickets(mask)
            }
            _ => Action::draw_hidden(),
        };

        self.engine.step(action);
        Ok(())
    }

    pub fn clone_game(&self) -> Self {
        self.clone()
    }
}

/// Batched Vectorized Environment in native multi-threaded Rust (releasing GIL).
#[pyclass]
pub struct PyVectorEnv {
    pub engines: Vec<GameEngine>,
    pub num_envs: usize,
}

#[pymethods]
impl PyVectorEnv {
    #[new]
    pub fn new(num_envs: usize, base_seed: u64) -> Self {
        let mut engines = Vec::with_capacity(num_envs);
        for i in 0..num_envs {
            engines.push(GameEngine::new(base_seed + (i as u64) * 1000));
        }
        PyVectorEnv { engines, num_envs }
    }

    pub fn reset_all(&mut self, py: Python<'_>, base_seed: u64) {
        py.allow_threads(|| {
            self.engines.par_iter_mut().enumerate().for_each(|(i, engine)| {
                engine.reset(base_seed + (i as u64) * 1000);
            });
        });
    }

    pub fn step_batch_sim(&mut self, py: Python<'_>, num_steps_per_env: usize) -> usize {
        let total_steps: usize = py.allow_threads(|| {
            self.engines
                .par_iter_mut()
                .map(|engine| {
                    let mut s = 0;
                    for _ in 0..num_steps_per_env {
                        if engine.state.is_game_over {
                            engine.reset(engine.state.rng_state[0] + 1);
                        }
                        let actions = engine.valid_actions();
                        if !actions.is_empty() {
                            engine.step(actions[0]);
                            s += 1;
                        }
                    }
                    s
                })
                .sum()
        });
        total_steps
    }
}
