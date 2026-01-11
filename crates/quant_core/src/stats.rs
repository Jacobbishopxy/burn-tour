//! Lightweight stats helpers (no external dependencies).
//!
//! These helpers are intentionally minimal; the point is to make the math obvious
//! while you learn, not to be maximally feature-complete.

/// Returns the arithmetic mean of `values`.
///
/// Returns `None` when `values` is empty.
///
/// # Examples
/// ```
/// use quant_core::stats::mean;
///
/// assert_eq!(mean(&[1.0, 2.0, 3.0]), Some(2.0));
/// assert_eq!(mean(&[]), None);
/// ```
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Returns the sample standard deviation (Bessel-corrected) of `values`.
///
/// Returns `None` when `values.len() < 2`.
///
/// # Examples
/// ```
/// use quant_core::stats::sample_std_dev;
///
/// // sample std dev for [1,2,3] is 1.0
/// assert_eq!(sample_std_dev(&[1.0, 2.0, 3.0]), Some(1.0));
/// ```
pub fn sample_std_dev(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean_value = mean(values)?;
    let sum_sq = values
        .iter()
        .map(|x| {
            let d = x - mean_value;
            d * d
        })
        .sum::<f64>();
    let variance = sum_sq / (values.len() as f64 - 1.0);
    Some(variance.sqrt())
}

