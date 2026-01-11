# Quant Metrics Basics (Returns → Equity → Drawdown → Sharpe)

## Goal

Implement and *understand* the basic metrics you will use constantly in research, backtests, and model evaluation.

## Key Concepts

- **Simple returns**: `r[t] = P[t] / P[t-1] - 1`
- **Equity curve**: start at 1.0 and compound returns: `E[t+1] = E[t] * (1 + r[t])`
- **Drawdown**: peak-to-trough decline on the equity curve (not on the raw returns)
- **Sharpe ratio**: risk-adjusted return proxy; very sensitive to return definition and sampling frequency

## Pitfalls (Finance + ML)

- **Look-ahead bias**: computing features/labels using future prices (most common failure mode)
- **Frequency mismatch**: daily returns + monthly risk-free rate = nonsense
- **Non-stationarity**: a “good Sharpe” in-sample often collapses out-of-sample
- **Ignoring costs**: turnover can turn a good model into a losing strategy

## References

- Any backtesting text: focus on return definitions, compounding, and drawdowns.
- Burn is not involved here; keep these utilities framework-agnostic.

## Code

- `crates/quant_core/src/time_series.rs`
- `crates/quant_lab/src/main.rs`
