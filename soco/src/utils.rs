//! Utilities.

use std::cmp::{max, min};

use crate::result::{Error, Result};

/// Safely asserts `pred`.
pub fn assert(pred: bool, error: Error) -> Result<()> {
    if pred {
        Ok(())
    } else {
        Err(error)
    }
}

/// Returns the fractional part of a float.
pub fn frac(x: f64) -> f64 {
    x - x.floor()
}

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
pub fn is_pow_of_2(x: i32) -> bool {
    x != 0 && x & (x - 1) == 0
}

/// Returns the `i`-th element of vector `xs` if present; `0.` otherwise.
pub fn faccess(xs: &Vec<f64>, i: i32) -> f64 {
    if i >= 0 && i < xs.len() as i32 {
        xs[i as usize]
    } else {
        0.
    }
}

/// Returns the `i`-th element of vector `xs` if present; `0` otherwise.
pub fn iaccess(xs: &Vec<i32>, i: i32) -> i32 {
    if i >= 0 && i < xs.len() as i32 {
        xs[i as usize]
    } else {
        0
    }
}
