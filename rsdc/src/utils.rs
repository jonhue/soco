//! Utilities.

use std::cmp::{max, min};

/// max{0, x}
pub fn ipos(x: i32) -> i32 {
    max(0, x)
}

/// max{0., x}
pub fn fpos(x: f64) -> f64 {
    if x > 0. {
        x
    } else {
        0.
    }
}

/// max{a, min{b, x}}
pub fn iproject(x: i32, a: i32, b: i32) -> i32 {
    max(a, min(b, x))
}

/// max{a, min{b, x}}
pub fn fproject(x: f64, a: f64, b: f64) -> f64 {
    let tmp = if b < x { b } else { x };
    if a > tmp {
        a
    } else {
        tmp
    }
}

/// Determines whether `x` is a power of `2`.
pub fn is_2pow(x: i32) -> bool {
    x != 0 && x & (x - 1) == 0
}
