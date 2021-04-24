//! Utilities.

use std::cmp::{max, min};

/// max{0, x}
pub fn ipos(x: i32) -> i32 {
    max(0, x)
}

/// max{a, min{b, x}}
pub fn iproject(x: i32, a: i32, b: i32) -> i32 {
    max(a, min(b, x))
}

/// Determines whether `x` is a power of `2`.
pub fn is_2pow(x: i32) -> bool {
    x != 0 && x & (x - 1) == 0
}
