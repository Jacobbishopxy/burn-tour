# Repository Guidelines

## Project Structure

- `Cargo.toml`: workspace definition (Rust edition 2024).
- `crates/quant_core`: reusable quant utilities (metrics, splits, validation, backtest primitives).
- `crates/quant_lab`: runnable scratchpad binaries and demos (intentionally “messy-on-purpose”).
- `crates/burn_lab`: Burn exercises; Burn deps are gated behind the `burn` feature.
- `docs/notes/`: short notes linked to runnable code (`docs/notes/00_template.md`).
- `plan/`: study plan and evidence log (`plan/burn_study_guide.md`, `plan/progress.md`).

## Build, Test, and Development Commands

- Run quant demo: `cargo run -p quant_lab`
- Run a specific demo: `cargo run -p quant_lab --bin 01_walk_forward_splits`
- Run Burn lab (enable Burn deps): `cargo run -p burn_lab --features burn`
- Run tests (preferred target): `cargo test -p quant_core`
- Format: `cargo fmt`
- Lint: `cargo clippy -p quant_core -p quant_lab` (optionally add `-p burn_lab --features burn`)

## Coding Style & Naming Conventions

- Rust: follow `rustfmt` defaults; keep public APIs in `quant_core` small and well-named.
- Binaries: use `crates/*/src/bin/NN_topic_name.rs` (two-digit prefix keeps an ordered curriculum).
- Notes: use `docs/notes/<topic>.md` and link the exact runnable file(s).
- Example code: add short comments explaining *why* (intent/pitfalls), not just what the line does; avoid duplicated comments (duplicated code across examples is OK for consistency).
- Every time you complete a runnable binary or test, record the exact command to run it in the related note under `docs/notes/` (or the most relevant `plan/*` entry).

## Testing Guidelines

- Prefer unit tests in `crates/quant_core/src/**` via `#[cfg(test)] mod tests { ... }`.
- Use integration tests when appropriate: `crates/quant_core/tests/<feature>_test.rs`.
- Add doc-tests for tiny examples when it improves readability (`/// ```rust ... ``` `).

## Commit & Pull Request Guidelines

- History currently uses a single, short, one-line summary (optionally with an emoji prefix). Keep commits similarly small and descriptive (e.g., `quant_core: add turnover metric`).
- PRs should include: what changed, how to run it (commands), and “evidence” updates when relevant (new runnable bin + note + checkbox update in `plan/progress.md`).

## Agent Notes (Codex)

- Avoid changing `Cargo.lock` unless you intentionally change dependencies/features.
- Keep changes scoped to one milestone at a time; update `docs/notes/` and `plan/progress.md` together with the code.
- After writing/modifying Rust code, run `cargo check` for the affected package/binary (e.g. `cargo check -p burn_lab --features burn --bin 02_device_movement`).
