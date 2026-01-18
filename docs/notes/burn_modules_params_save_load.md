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

## Layer Selection (Quick Rubric)

- **Input shape & invariances**:
  - Images / spatial grids → `Conv2d` + pooling (translation bias).
  - Sequences / time series → `Conv1d`, `RNN/GRU/LSTM`, or `Transformer` (order matters).
  - Tabular / fixed features → `Linear`/MLP (no strong spatial bias).
- **Context length**:
  - Short local patterns → 1D conv.
  - Long-range dependencies → attention / transformer.
- **Data size vs capacity**:
  - Small data → simpler models (linear/MLP, fewer layers).
  - Large data → deeper/wider nets, attention, embeddings.
- **Stability & batching**:
  - Small batch sizes → prefer `LayerNorm`/`RMSNorm` over `BatchNorm`.
- **Latency/compute limits**:
  - Attention is O(n^2) in sequence length; conv/RNN can be cheaper.

Typical starting points in Burn:

- Tabular → `Linear` → `ReLU` → `Linear` (optionally `Dropout`/`LayerNorm`).
- Time series → small `Conv1d` stack or GRU; move to Transformer if long-range matters.
- Text → `Embedding` + `TransformerEncoder` (or a simpler RNN).
- Images → conv blocks.

## Code

- `crates/burn_lab/src/bin/05_modules_params_save_load.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 05_modules_params_save_load`
