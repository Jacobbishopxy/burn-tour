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
- The batcher is the DataLoader "collate" step:
  - it receives `Vec<LinSample>` and builds a single `LinBatch` of tensors
  - it flattens per-item `x`/`y` values into contiguous buffers and shapes them as
    `[batch_size, 2]` and `[batch_size, 1]` on the target device
  - this keeps the model API batched and avoids per-sample overhead in the loop

## Why Burn Calls It `Learner`

Burn's `Learner` is the high-level training driver: it wires together model + optimizer +
LR schedule + metrics + checkpoints, then runs the train/valid loop. The name
is a general ML term for a learning algorithm/agent, but it isn't universal; many
frameworks call the same role "Trainer" or "Estimator."

## Graph (High-Level Flow)

```mermaid
flowchart TD
  DS[LinDataset] --> DL["DataLoader (Batcher)"]
  DL --> B[LinBatch]
  B --> TS["TrainStep (Autodiff backend)"]
  TS --> BW["loss.backward()"]
  BW --> GR[Gradients]
  GR --> OPT["Optimizer step (Learner)"]
  B --> VS["ValidStep (non-autodiff backend)"]
  VS --> MET[Metrics / logging]
  LB[LearnerBuilder\nmodel + optimizer + metrics + checkpoints + epochs] -.-> TS
  LB -.-> VS
```

## Graph (Struct Usage + Dependencies)

```mermaid
flowchart TD
  LS["LinSample\nx: [f32; 2]\ny: [f32; 1]"]
  LD[LinDataset\nsamples: Vec<LinSample>]
  LB[LinBatch\nx: Tensor\ny: Tensor]
  LBT[LinBatcher]
  LBTt["LinBatcher (train)"]
  LBTv["LinBatcher (valid)"]
  LR[LinReg\nlinear: Linear]
  LRI["LinReg::new(&device_train)"]
  DLB[DataLoaderBuilder]
  DLT["DataLoader (train)"]
  DLV["DataLoader (valid)"]
  LRB[LearnerBuilder]
  LRN[Learner]
  OPT["Adam (optimizer)"]
  MET[LossMetric]
  TB["TrainB\nAutodiff<NdArray>"]
  VB["ValidB\nNdArray"]
  DT["device_train"]
  DV["device_valid"]
  DIR["Checkpoint dir\ntarget/burn_lab/07_training_learner"]
  TM["trained.model\nValidB"]

  LD -->|contains| LS
  LD -->|implements| DS[Dataset<LinSample>]
  LBTt -->|batches Vec<LinSample>| LB
  LBTt -->|consumes| LS
  LBTv -->|batches Vec<LinSample>| LB
  LBTv -->|consumes| LS
  DLB -->|builds| DLT
  DLB -->|builds| DLV
  DLT -->|uses| LBTt
  DLV -->|uses| LBTv
  DLT -->|yields| LB
  DLV -->|yields| LB
  TB -->|backend for| LR
  DT -->|used by| LRI
  LRI -->|creates| LR
  TB -->|batches on| DLT
  VB -->|batches on| DLV
  VB -->|metric backend| MET
  DT -->|device for| TB
  DV -->|device for| VB
  LR -->|forward uses| LB
  LR -->|implements| TS[TrainStep<LinBatch, RegressionOutput>]
  LR -->|implements| VS[ValidStep<LinBatch, RegressionOutput>]
  LRB -->|configures| LRN
  LRB -->|uses| OPT
  LRB -->|uses| MET
  LRB -->|uses| LR
  LRB -->|writes to| DIR
  LRN -->|fit uses| DLT
  LRN -->|fit uses| DLV
  LRN -->|produces| TM
  LRN -->|writes to| DIR
```

## Code

- `crates/burn_lab/src/bin/07_training_learner.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 07_training_learner`
