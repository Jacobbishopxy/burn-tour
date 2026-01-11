//! Quant utilities for research and backtests.
//!
//! Design goals:
//! - Keep functions simple and explicit.
//! - Prefer "boring" primitives (`f64`, slices) for clarity while studying.
//! - Document assumptions to avoid common finance bugs (look-ahead bias, scale issues).
//!
//! You can run the demo binary in `crates/quant_lab`:
//! - `cargo run -p quant_lab`

pub mod stats;
pub mod time_series;
pub mod validation;
pub mod portfolio;
