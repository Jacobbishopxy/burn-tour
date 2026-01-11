//! `quant_lab` is your runnable scratchpad for quant concepts.
//!
//! The expectation is that you keep this crate messy-on-purpose:
//! - Small experiments you can run quickly.
//! - Lots of doc links and comments.
//! - Clear "evidence" that you understood a topic (a working example + notes).

use quant_core::time_series::{equity_curve, max_drawdown, sharpe_ratio, simple_returns};

fn main() {
    // Example "price series" (toy).
    let prices = [100.0, 101.0, 99.5, 102.0, 101.0];

    let returns = simple_returns(&prices);
    let equity = equity_curve(&returns);

    println!("prices  = {prices:?}");
    println!("returns = {returns:?}");
    println!("equity  = {equity:?}");

    let dd = max_drawdown(&equity).unwrap_or(0.0);
    let sr = sharpe_ratio(&returns, 0.0, 252.0).unwrap_or(0.0);

    println!("max_drawdown = {dd:.4}");
    println!("sharpe_ratio = {sr:.4}");
}

