# Burn Configs + Reproducible Hyperparams

## Goal

Make experiments repeatable by:

- storing hyperparams in a `Config` struct (JSON)
- seeding the backend so randomized initialization is reproducible

## Mental Model

- A `Config` is the “source of truth” for *how to build* a model/run, not the model weights themselves.
- If initialization uses randomness (common defaults), call `Backend::seed(&device, seed)` before building to get repeatable starts.

## TinyNetConfig Graph

```mermaid
flowchart LR
    Seed["seed (u64)"] --> SeedCall["Backend::seed(&device, seed)"]

    In["Input Tensor<br/>[batch, d_in]"] --> L1["Linear l1<br/>d_in -> d_hidden"]
    L1 --> Act["ReLU"]
    Act --> L2["Linear l2<br/>d_hidden -> d_out"]
    L2 --> Out["Output Tensor<br/>[batch, d_out]"]

    DIn["d_in (usize)"] -.-> L1
    DHid["d_hidden (usize)"] -.-> L1
    DHid -.-> L2
    DOut["d_out (usize)"] -.-> L2
```

## Code

- `crates/burn_lab/src/bin/06_configs_repro_hparams.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 06_configs_repro_hparams`
