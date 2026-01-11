# burn-tour (workspace)

This repo is a Rust workspace designed to support an evidence-based study plan for:
- Burn (deep learning in Rust)
- Quant-investment engineering fundamentals (data hygiene, evaluation, backtesting)

Start here:
- Study plan: `plan/burn_study_guide.md`
- Progress log: `plan/progress.md`
- Structure options: `plan/project_structure.md`

## Quick Commands

- Run quant demo: `cargo run -p quant_lab`
- Run walk-forward split demo: `cargo run -p quant_lab --bin 01_walk_forward_splits`
- Run tests (quant utilities): `cargo test -p quant_core`

Burn dependencies are optional in this workspace:
- Enable when ready: `cargo run -p burn_lab --features burn`

