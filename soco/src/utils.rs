//! Utilities.

use crate::result::{Failure, Result};
use crate::value::Value;
use num::NumCast;
use rand::{thread_rng, Rng};

/// Safely asserts `pred`.
pub fn assert(pred: bool, failure: Failure) -> Result<()> {
    if pred {
        Ok(())
    } else {
        Err(failure)
    }
}

/// Returns the fractional part of a float.
pub fn frac(x: f64) -> f64 {
    x - x.floor()
}

/// max{0, x}
pub fn max<T>(x: T, y: T) -> T
where
    T: Value,
{
    if x > y {
        x
    } else {
        y
    }
}

/// max{0, x}
pub fn min<T>(x: T, y: T) -> T
where
    T: Value,
{
    if x > y {
        y
    } else {
        x
    }
}

/// max{0, x}
pub fn pos<T>(x: T) -> T
where
    T: Value,
{
    max(NumCast::from(0).unwrap(), x)
}

/// max{a, min{b, x}}
pub fn project<T>(x: T, a: T, b: T) -> T
where
    T: Value,
{
    max(a, min(b, x))
}

/// Determines whether `x` is a power of `2`.
pub fn is_pow_of_2(x: i32) -> bool {
    x != 0 && x & (x - 1) == 0
}

/// Returns the `i`-th element if it exists.
pub fn access<T>(xs: &Vec<T>, i: i32) -> Option<&T> {
    if i >= 0 && i < xs.len() as i32 {
        Some(&xs[i as usize])
    } else {
        None
    }
}

/// Computes the sum of bounds across all dimensions.
pub fn total_bound<T>(bounds: &Vec<T>) -> T
where
    T: Value,
{
    let mut result: T = NumCast::from(0).unwrap();
    for &b in bounds {
        result = result + b;
    }
    result
}

/// Randomly samples a uniform value in `[a,b]`.
pub fn sample_uniform(a: f64, b: f64) -> f64 {
    let mut rng = thread_rng();
    rng.gen_range(a..=b)
}
