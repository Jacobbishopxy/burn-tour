# Learning Progress Log

Edit this file as you study. Keep it short and evidence-based.

## How to Use

For each checkbox you complete, add:

- 1 link to runnable code (a file path in this repo)
- 1 link to a note in `docs/notes/`
- 2–3 sentences: what you learned + what confused you

Rule: the checkbox state (`[ ]` / `[x]`) is only edited by the user; assistants can add links/notes but should not toggle checkboxes.

## Milestones (Study Line)

### Setup

- [x] Workspace created (`plan/project_structure.md`)
- [x] Quant starter crate runs (`crates/quant_lab/src/main.rs`)

### Burn + Training + Quant Engineering

- [x] 1. Burn tensors + devices ([code](../docs/notes/burn_tensor_basics.md))
- [x] 2. Backends mental model ([code](../crates/burn_lab/src/bin/03_backends_mental_model.rs), [note](../docs/notes/burn_backends_mental_model.md)) — Learned that `Backend` is a compile-time choice that fixes `Tensor` storage/kernels and the concrete `Device` type you pass around.
- [x] 3. Autodiff debugging checklist ([code](../crates/burn_lab/src/bin/04_autodiff_debugging_checklist.rs), [note](../docs/notes/burn_autodiff_debugging_checklist.md)) — Learned that you need both an autodiff backend (`Autodiff<...>`) and `require_grad()` on leaf tensors to get gradients. The most common “why is grad None?” failure mode so far is accidentally breaking the graph with `detach()`/inner conversions while printing/debugging.
- [x] 4. Modules + parameters + save/load ([code](../crates/burn_lab/src/bin/05_modules_params_save_load.rs), [note](../docs/notes/burn_modules_params_save_load.md)) — Learned that Burn “models” are plain Rust structs with `#[derive(Module)]` and nested sub-modules (like `nn::Linear`) already contain `Param<Tensor<...>>` parameters. Save/load is record-based: you build the architecture, then `load_file(...)` swaps in the saved parameter tensors (verified by identical forward outputs).
- [ ] 5. Configs + reproducible hyperparams ([code](../crates/burn_lab/src/bin/06_configs_repro_hparams.rs), [note](../docs/notes/burn_configs_repro_hparams.md)) — Learned that Burn’s `#[derive(Config)]` gives you a JSON-serializable hyperparam struct (including built-in `save`/`load`). Seeding the backend (`Backend::seed`) before initialization makes randomized layer init reproducible (verified by identical forward outputs across re-inits with the same seed).
- [ ] 6. Training (high-level `Learner`) ([code](../crates/burn_lab/src/bin/07_training_learner.rs), [note](../docs/notes/burn_training_learner.md)) — Learned that with `LearnerBuilder` you only implement `TrainStep`/`ValidStep` and the trainer handles epochs, metrics logging, and checkpoint wiring. Also learned that metric items are synced off-thread via `ItemLazy` (e.g., `RegressionOutput`), so you can keep the training loop from blocking on tensor reads.
- [ ] 7. Training (custom loop) ([code](../crates/burn_lab/src/bin/08_training_custom_loop.rs), [note](../docs/notes/burn_training_custom_loop.md)) — Learned the “bare metal” loop: forward → loss → `backward()` → `optimizer.step(...)`, and that Burn optimizers typically return an updated module (no in-place updates). Also learned that calling `into_data()` during training is a sync point, so you want to keep it out of the inner loop unless you’re actively debugging.
- [ ] 8. Baseline quant metrics expanded (code + note)
- [ ] 9. Time-series validation split + tests (code + note)
- [ ] 10. Labeling & targets tied to trading actions (note + a small example)
- [ ] 11. Model families + selection rubric (note + at least 2 prototypes)
- [ ] 12. Toy backtest with costs (code + note)
- [ ] 13. Risk + sizing constraints (code + note)
- [ ] 14. Inference contract + monitoring checklist (note + a small example)

## Session Log (Append-Only)

Add newest entries at the top.

### YYYY-MM-DD

- Focus:
- Evidence:
  - Code:
  - Notes:
- What I learned:
- Open questions:
- Next session:
