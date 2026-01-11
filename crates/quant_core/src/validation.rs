//! Validation utilities focused on time-series workflows.
//!
//! In finance, random shuffles almost always create leakage. Prefer walk-forward splits.

use core::ops::Range;

/// Produces walk-forward train/test splits over an index range `[0, len)`.
///
/// Each split is:
/// - `train = [start, start + train_len)`
/// - `test  = [start + train_len, start + train_len + test_len)`
///
/// After emitting a split, `start` advances by `step`.
///
/// Returns an empty vector when any window is invalid or no full split fits.
///
/// # Examples
/// ```
/// use quant_core::validation::walk_forward_splits;
///
/// let splits = walk_forward_splits(10, 4, 2, 2);
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].0, 0..4);
/// assert_eq!(splits[0].1, 4..6);
/// assert_eq!(splits[1].0, 2..6);
/// assert_eq!(splits[1].1, 6..8);
/// assert_eq!(splits[2].0, 4..8);
/// assert_eq!(splits[2].1, 8..10);
/// ```
pub fn walk_forward_splits(
    len: usize,
    train_len: usize,
    test_len: usize,
    step: usize,
) -> Vec<(Range<usize>, Range<usize>)> {
    if len == 0 || train_len == 0 || test_len == 0 || step == 0 {
        return Vec::new();
    }
    if train_len + test_len > len {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut start = 0_usize;
    while start + train_len + test_len <= len {
        let train = start..(start + train_len);
        let test = (start + train_len)..(start + train_len + test_len);
        out.push((train, test));
        start += step;
    }
    out
}

