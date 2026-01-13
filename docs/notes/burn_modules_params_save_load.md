# Burn Modules, Params, Save/Load

## Goal

Get comfortable with Burn’s core “model as a Rust struct” workflow:

- define a `Module` (and nested sub-modules)
- understand where parameters live (`Param<Tensor<...>>`)
- save/load a model with a recorder

## Key Takeaways

- `#[derive(Module)]` generates the plumbing to:
  - traverse parameters (`num_params`, `visit`, `map`)
  - serialize/deserialize via a generated `Record`
  - save/load with `save_file` / `load_file`
- Loading is “architecture + weights”: you typically construct the module shape, then `load_file(...)` replaces the parameter tensors from the record.

## Code

- `crates/burn_lab/src/bin/05_modules_params_save_load.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 05_modules_params_save_load`

