# Burn Training with `Learner` (High-Level)

## Goal

Get a working mental model for Burn’s high-level training stack:

- implement `TrainStep` / `ValidStep` on a `Module`
- wire everything together with `LearnerBuilder`
- get metrics + logs + checkpoints “for free”

## What to Look For

- `TrainStep::step` returns both:
  - gradients (from `loss.backward()`)
  - a metric-friendly output item (`RegressionOutput`), which the learner syncs off-thread
- Validation runs on the inner backend (no autodiff) via `ValidStep` on `InnerModule`

## Code

- `crates/burn_lab/src/bin/07_training_learner.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 07_training_learner`

