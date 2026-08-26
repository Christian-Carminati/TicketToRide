//! ttr_core: High-performance native Rust core and bitboard game engine for TicketToRide RL Lab.

pub mod board;
pub mod dsu;
pub mod game;
pub mod py_bindings;
pub mod rules;
pub mod state;
pub mod types;

use pyo3::prelude::*;
use py_bindings::{PyGame, PyVectorEnv};

#[pymodule]
fn ttr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyVectorEnv>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
