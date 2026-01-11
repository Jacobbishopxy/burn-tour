//! Walk-forward split demo.
//!
//! Why it matters in quant:
//! - Random train/test splits usually leak future information.
//! - Walk-forward makes the time direction explicit.

use quant_core::validation::walk_forward_splits;

fn main() {
    let len = 30;
    let train = 10;
    let test = 5;
    let step = 5;

    let splits = walk_forward_splits(len, train, test, step);
    for (i, (tr, te)) in splits.iter().enumerate() {
        println!("{i}: train={tr:?} test={te:?}");
    }
}

