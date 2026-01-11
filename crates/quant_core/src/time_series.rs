//! Time-series utilities commonly used in quant research.
//!
//! Important caveat: these functions operate on already-clean numeric series.
//! In real pipelines you must handle missing data, corporate actions, and
//! calendar alignment explicitly.

use crate::stats::{mean, sample_std_dev};

/// Computes simple (percentage) returns from a price series.
///
/// Output length is `prices.len() - 1`:
/// - `r[t] = prices[t+1] / prices[t] - 1`
///
/// Returns an empty vector when `prices.len() < 2`.
///
/// # Examples
/// ```
/// use quant_core::time_series::simple_returns;
///
/// let prices = [100.0, 110.0, 104.5];
/// let r = simple_returns(&prices);
/// assert!((r[0] - 0.10).abs() < 1e-12);
/// assert!((r[1] - (-0.05)).abs() < 1e-12);
/// ```
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]) - 1.0)
        .collect()
}

/// Computes an equity curve from period returns, starting at 1.0.
///
/// Output length is `returns.len() + 1` (including the initial 1.0).
///
/// # Examples
/// ```
/// use quant_core::time_series::equity_curve;
///
/// let equity = equity_curve(&[0.1, -0.05]);
/// assert!((equity[0] - 1.0).abs() < 1e-12);
/// assert!((equity[1] - 1.1).abs() < 1e-12);
/// assert!((equity[2] - 1.045).abs() < 1e-12);
/// ```
pub fn equity_curve(returns: &[f64]) -> Vec<f64> {
    let mut equity = Vec::with_capacity(returns.len() + 1);
    equity.push(1.0);
    for r in returns {
        let next = equity.last().copied().unwrap_or(1.0) * (1.0 + r);
        equity.push(next);
    }
    equity
}

/// Computes the maximum drawdown for an equity curve.
///
/// The result is returned as a positive fraction (e.g. 0.22 means -22% peak-to-trough).
/// Returns `None` when `equity` is empty.
///
/// # Examples
/// ```
/// use quant_core::time_series::max_drawdown;
///
/// // Peak 1.2 -> trough 0.9 => drawdown 25%
/// let dd = max_drawdown(&[1.0, 1.2, 0.9, 1.1]).unwrap();
/// assert!((dd - 0.25).abs() < 1e-12);
/// ```
pub fn max_drawdown(equity: &[f64]) -> Option<f64> {
    if equity.is_empty() {
        return None;
    }

    let mut peak = equity[0];
    let mut max_dd = 0.0_f64;
    for &x in equity.iter() {
        if x > peak {
            peak = x;
            continue;
        }
        if peak > 0.0 {
            let dd = (peak - x) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    Some(max_dd)
}

/// Computes an annualized Sharpe ratio from period returns.
///
/// - `risk_free_rate` is the per-period risk-free rate (match the returns frequency).
/// - `periods_per_year` is 252 for daily, 52 for weekly, 12 for monthly, etc.
///
/// Returns `None` when `returns.len() < 2` or the return volatility is 0.
///
/// # Examples
/// ```
/// use quant_core::time_series::sharpe_ratio;
///
/// let r = [0.01, -0.005, 0.002, 0.004];
/// let sr = sharpe_ratio(&r, 0.0, 252.0).unwrap();
/// assert!(sr.is_finite());
/// ```
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> Option<f64> {
    if returns.len() < 2 || periods_per_year <= 0.0 {
        return None;
    }

    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free_rate).collect();
    let mean_excess = mean(&excess)?;
    let vol = sample_std_dev(&excess)?;
    if vol == 0.0 {
        return None;
    }
    Some(mean_excess / vol * periods_per_year.sqrt())
}
