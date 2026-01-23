# Burn + Quant-Investment Engineer Study Plan (Phase 2+)

Goal: build deep understanding of Burn core concepts + training, then apply them to the workflows a quant-investment engineer actually needs: data hygiene, model selection, evaluation, backtesting, and production-minded engineering.

This guide intentionally emphasizes **examples**, **documentation**, and **commented code** as “evidence” of learning.

## How This Plan Works (Evidence-Based)

For every step, produce 3 artifacts:

1. **Runnable code** (small, focused): `crates/*/src/bin/...` or `crates/*/src/...`
2. **A short note** (what/why/pitfalls): `docs/notes/<topic>.md` (start from `docs/notes/00_template.md`)
3. **A checked box + 2–3 sentences** in your progress log: `plan/progress.md`

## Workspace (You Already Have This)

See structure options in `plan/project_structure.md`. Current repo implements Option A:

- `crates/burn_lab`: Burn exercises (enable deps when needed)
- `crates/quant_core`: reusable quant utilities (metrics, splits, backtest primitives)
- `crates/quant_lab`: runnable quant scratchpad
- `docs/notes`: notes linked to code
- `plan`: study plan + progress tracking

## Study Line (Ordered Milestones)

You can treat each item below as a “ticket”: implement it, document it, then mark it done in `plan/progress.md`.

1. **Burn tensors + devices**: shapes, dtypes, broadcasting, `Device`, moving tensors between devices. (Ref: *Building Blocks > Tensor*)
2. **Backends mental model**: what changes across `NdArray` / `Wgpu` / others, and what shouldn’t. (Ref: *Building Blocks > Backend*)
3. **Autodiff**: `ADBackendDecorator`, gradient flow, “why no grad?” debugging checklist. (Ref: *Building Blocks > Autodiff*)
4. **Modules**: `#[derive(Module)]`, parameter ownership, save/load, initialization patterns. (Ref: *Building Blocks > Module*)
5. **Configs**: `#[derive(Config)]`, reproducible hyperparameters, serialization of configs. (Ref: *Building Blocks > Config*)
6. **Training (high-level)**: dataset → dataloader → model → loss → optimizer → metrics → checkpointing. (Ref: *Basic Workflow* & *Building Blocks > Learner*)
7. **Training (low-level)**: write a tiny custom loop (iterate epochs/batches -> forward -> backward -> optim step). (Ref: *Custom Training Loop*)
8. **Baseline quant metrics**: returns, equity curve, drawdown, Sharpe (implemented in `quant_core`; extend it).
9. **Time-series validation**: walk-forward split + leakage checklist; write tests for your split logic.
10. **Labeling & targets**: define targets that match trading actions (classification vs regression vs ranking).
11. **Model families + selection**: choose baselines first; only then add deep sequence models.
12. **Backtest essentials (toy but correct)**: signal → positions → PnL with costs; avoid look-ahead.
13. **Risk + sizing**: exposure caps, leverage, turnover, drawdown controls.
14. **Inference & deployment shape**: model packaging, feature parity, deterministic inference, monitoring. (Ref: *Basic Workflow > Inference*, *Importing Models*)

## Phase 2: Burn Core Concepts (The “What”)

**Focus Chapters:** *Building Blocks* (Tensor, Backend, Autodiff, Module, Config)

### What you must be able to explain (in your own words)

- Why Burn makes `Tensor<B, D>` generic over a `Backend` (swapping WGPU/Candle/LibTorch/NdArray).
- When you need autodiff (`ADBackendDecorator`) and when you explicitly do *not* want it (inference speed/footprint).
- How shapes flow through a forward pass (and how you debug shape mismatches).

### What you must implement (code + notes)

- `docs/notes/burn_tensor_basics.md`
- Add at least 2 runnable examples under `crates/burn_lab/src/bin/` (one per topic), heavily commented:
  - tensor creation + reshape + matmul (explicit types vs inferred).
  - device movement + a small benchmarking note (even if informal).

## Phase 3: Training Framework (The “How”)

**Focus Chapters:** *Basic Workflow*, *Building Blocks* (Learner, Metric, Dataset), *Custom Training Loop*

### Core training workflow (Burn-specific)

- Datasets and dataloaders (incl. shuffling and deterministic seeds).
- `Learner` configuration: metrics, checkpoints (`NamedMpkFileRecorder`), early stopping.
- Optimizers, schedulers, and common failure modes (divergence, NaNs, exploding gradients).
- **Deep Dive (Custom Loop)**: Explicitly calling `loss.backward()`, mapping gradients to params (`GradientsParams`), and `optim.step(lr, model, grads)`. This is often required for RL or non-standard quant objectives.

### Deliverables 1

- `docs/notes/burn_training_framework.md` (Contrast `Learner` vs Custom Loop).
- One tiny regression problem end-to-end (synthetic data is fine).
- One classification problem end-to-end (synthetic is fine; MNIST later if you want).

## Phase 4: Quant Data + Validation (The “Finance Reality”)

### Non-negotiables for quant ML

- Data leakage taxonomy: look-ahead, survivorship, delayed availability, corporate actions, overlapping labels.
- Time-series splits: walk-forward, embargo/purge concepts (know *why*, even if you don’t implement all variants).
- Evaluation alignment: the ML metric must map to PnL behavior (not just accuracy).

### Deliverables 2

- Extend `crates/quant_core` with:
  - `walk_forward_split(...)` (and tests).
  - `turnover(...)` (and tests).
  - at least 1 “don’t do this” example (a leaky split) documented in `docs/notes/`.
- `docs/notes/finance_data_leakage.md`

## Phase 5: Common Model Families + Selection (Quant View)

### Model selection rubric (keep it simple)

- Start with: linear / ridge / lasso style baselines (interpretability, stability).
- Then: tree/boosting-style baselines (if/when you use another toolchain later).
- Then: deep learning (Burn) when you have enough data and a clear reason:
  - MLPs for cross-sectional signals.
  - 1D-CNNs / TCNs for local temporal patterns.
  - RNN/LSTM/GRU for sequential dependencies (often outperformed by TCN/Transformer, but learn the basics).
  - Transformers for long-range context (costly; be explicit about why).

### Deliverables 3

- `docs/notes/model_selection_quant.md` (include: “when *not* to use deep learning”).
- Implement at least 1 baseline signal model and 1 deep model prototype as runnable examples.

## Phase 6: Backtesting + Production Concerns (Engineer View)

**Focus Chapters:** *Importing Models* (ONNX/PyTorch), *Performance*

### Backtest essentials (minimum viable correctness)

- A strict “features at time t” rule (no future columns).
- Transaction costs + slippage placeholders (even if coarse).
- Position sizing and constraints (avoid unconstrained leverage).

### Production essentials (minimum viable engineering)

- **Importing Models**:
  - `burn-import` crate: Compiling ONNX models to Rust code (for speed/portability).
  - `PyTorchFileRecorder`: Loading weights directly from `.pt` files (e.g., from Python research).
- **Performance**:
  - Kernel Fusion (`burn-fusion`): How it groups operations to reduce memory bandwidth.
  - Async Execution: Non-blocking tensor operations.
- Reproducibility: deterministic seeds, config snapshots.
- Inference contract: feature ordering, scaling, missing data policy.

### Deliverables 4

- `docs/notes/toy_backtest_engine.md`
- A toy strategy backtest end-to-end (small but correct) + a postmortem note (“what broke and why”).

## References (Start Here)

- Burn Book: <https://burn.dev/books/burn>
- Burn docs.rs: <https://docs.rs/burn>
- Burn examples: <https://github.com/tracel-ai/burn/tree/main/examples>
