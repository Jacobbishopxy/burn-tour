//! Portfolio helpers (position sizing and turnover).

/// Computes one-step turnover between two weight vectors.
///
/// Common definition: `0.5 * sum_i |w_next[i] - w_prev[i]|`.
/// - The `0.5` factor is used because buys and sells both contribute.
/// - Inputs are assumed to be portfolio weights (not notionals).
///
/// Returns `None` if the vectors have different lengths or are empty.
///
/// # Examples
/// ```
/// use quant_core::portfolio::turnover;
///
/// // Shift 10% from asset A to B => turnover = 10%
/// let prev = [0.6, 0.4];
/// let next = [0.5, 0.5];
/// let t = turnover(&prev, &next).unwrap();
/// assert!((t - 0.1).abs() < 1e-12);
/// ```
pub fn turnover(prev: &[f64], next: &[f64]) -> Option<f64> {
    if prev.is_empty() || prev.len() != next.len() {
        return None;
    }

    let sum_abs = prev
        .iter()
        .zip(next.iter())
        .map(|(a, b)| (b - a).abs())
        .sum::<f64>();
    Some(0.5 * sum_abs)
}
