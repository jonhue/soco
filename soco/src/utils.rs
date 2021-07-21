//! Utilities.

use crate::result::{Failure, Result};
use num::{Num, NumCast};
use rand::{thread_rng, Rng};
use std::iter::Sum;

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

/// max{x, y}
pub fn max<T>(x: T, y: T) -> T
where
    T: PartialOrd,
{
    if x > y {
        x
    } else {
        y
    }
}

/// min{x, y}
pub fn min<T>(x: T, y: T) -> T
where
    T: PartialOrd,
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
    T: NumCast + PartialOrd,
{
    max(NumCast::from(0).unwrap(), x)
}

/// max{a, min{b, x}}
pub fn project<T>(x: T, a: T, b: T) -> T
where
    T: PartialOrd,
{
    max(a, min(b, x))
}

/// Determines whether `x` is a power of `2`.
pub fn is_pow_of_2(x: i32) -> bool {
    x != 0 && x & (x - 1) == 0
}

/// Returns the `i`-th element if it exists.
pub fn access<T>(xs: &Vec<T>, i: i32) -> Option<&T> {
    if 1 <= i && i <= xs.len() as i32 {
        Some(&xs[i as usize - 1])
    } else {
        None
    }
}

/// Randomly samples a uniform value in `[a,b]`.
pub fn sample_uniform(a: f64, b: f64) -> f64 {
    let mut rng = thread_rng();
    rng.gen_range(a..=b)
}

/// Move `t` such that the time scale begins at `t_start`.
pub fn shift_time(t: i32, t_start: i32) -> i32 {
    t + t_start - 1
}

/// Move `t` from a time scale beginning at `t_start` to a time scale beginning at `1`.
pub fn unshift_time(t: i32, t_start: i32) -> i32 {
    t - t_start + 1
}

/// Mean of a vector.
pub fn mean<T>(xs: Vec<T>) -> T
where
    T: Copy + Num + NumCast + Sum,
{
    assert!(!xs.is_empty());
    let n = NumCast::from(xs.len()).unwrap();
    xs.into_iter().sum::<T>() / n
}
