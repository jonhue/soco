//! Utilities.

use num::{Num, NumCast};

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
pub fn pos<T: NumCast + PartialOrd>(x: T) -> T {
    let l = NumCast::from(0).unwrap();
    if x > l {
        x
    } else {
        l
    }
}

/// max{a, min{b, x}}
pub fn project<T: NumCast + PartialOrd>(x: T, a: T, b: T) -> T {
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

/// Returns the `i`-th element of vector `xs` if present; `def` otherwise.
pub fn access<T: Clone>(xs: &Vec<T>, i: i32, def: T) -> T {
    if i >= 0 && i < xs.len() as i32 {
        xs[i as usize].clone()
    } else {
        def
    }
}

/// Computes the sum of bounds across all dimensions.
pub fn total_bound<T>(bounds: &Vec<T>) -> T
where
    T: Copy + Num + NumCast,
{
    let mut result: T = NumCast::from(0).unwrap();
    for &b in bounds {
        result = result + b;
    }
    result
}
