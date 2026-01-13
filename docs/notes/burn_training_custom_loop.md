# Burn Training with a Custom Loop

## Goal

Understand the “manual” training mechanics that `Learner` abstracts away:

- forward pass → compute loss → `backward()` → optimizer step
- switch to `model.valid()` for inference/validation (inner backend, no autodiff)

## Key Takeaways

- Burn optimizers usually return an updated module; the training loop often looks like `model = optim.step(lr, model, grads)`.
- Reading tensors (`into_data`) is a sync point; do it sparingly in tight loops.

## Code

- `crates/burn_lab/src/bin/08_training_custom_loop.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 08_training_custom_loop`

