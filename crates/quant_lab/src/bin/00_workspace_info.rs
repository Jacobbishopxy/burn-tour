//! Workspace info helper.
//!
//! This exists mostly to keep the repository root clean (no `./src` folder),
//! while still having a quick “what is this repo?” entrypoint you can run.

fn main() {
    println!("burn-tour workspace");
    println!("- Study plan: `plan/burn_study_guide.md`");
    println!("- Progress log: `plan/progress.md`");
    println!("- Run quant demo: `cargo run -p quant_lab`");
    println!("- Run Burn lab (when ready): `cargo run -p burn_lab --features burn`");
}

