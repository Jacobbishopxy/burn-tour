# Project Structure Options

Pick one structure and stick to it for at least 2–4 weeks (avoid churn while learning).

## Option A (Recommended): Cargo workspace + labs + docs

Good when you want clean separation between reusable libraries and messy experiments.

```txt
.
├─ crates/
│  ├─ burn_lab/      # Burn exercises (deep learning)
│  ├─ quant_core/    # Reusable quant utilities (metrics, splits, backtest primitives)
│  └─ quant_lab/     # Runnable scratchpad for quant experiments
├─ docs/notes/       # Short notes linked to code
└─ plan/             # Study plan + progress tracking
```

Pros: modular, easy to run examples, easy to keep notes close to code.
Cons: slightly more Cargo/workspace overhead.

## Option B: Single crate + modules + examples/

```txt
.
├─ src/              # modules: burn/, data/, models/, backtest/
└─ examples/         # runnable snippets (cargo run --example ...)
```

Pros: simplest Cargo setup.
Cons: tends to become a “junk drawer” over time; hard to reuse code cleanly.

## Option C: Research vs production split

```txt
.
├─ crates/
│  ├─ research/      # notebooks-like experiments (fast iteration, looser standards)
│  └─ prod/          # stricter library + CI + reproducibility
└─ docs/
```

Pros: mirrors real quant teams; good for long-term.
Cons: more process than you need at the start.
