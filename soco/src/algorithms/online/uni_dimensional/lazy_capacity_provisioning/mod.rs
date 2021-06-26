//! Lazy Capacity Provisioning.

use crate::value::Value;
use num::NumCast;

/// Last two lower and upper bound from some time `t` (in order).
#[derive(Clone, Debug)]
pub struct Memory<T>
where
    T: Value,
{
    pub lower: (Option<T>, T),
    pub upper: (Option<T>, T),
}

/// Finds a valid reference time and initial condition to base the optimization
/// on (alternatively to time `0`).
pub fn find_initial_time<T>(final_t: i32, ms: &Vec<Memory<T>>) -> (i32, T)
where
    T: Value,
{
    for t in (2..final_t).rev() {
        let m = &ms[t as usize - 1];
        if is_valid_initial_time(&m) {
            return (t, m.upper.0.unwrap());
        }
    }

    (0, NumCast::from(0.).unwrap())
}

/// Returns `true` if the time `t` with current bounds `m` and previous bounds
/// (at `t - 1`) `prev_m` may be used as reference time.
fn is_valid_initial_time<T>(Memory { lower, upper }: &Memory<T>) -> bool
where
    T: Value,
{
    upper.1 < upper.0.unwrap() || lower.1 > lower.0.unwrap()
}

/// Shifts the memory to include new lower bound `l` and upper bound `u`.
fn new_memory<T>(ms: &Vec<Memory<T>>, l: T, u: T) -> Memory<T>
where
    T: Value,
{
    let (prev_l, prev_u) = if ms.is_empty() {
        (None, None)
    } else {
        let m = &ms[ms.len() - 1];
        (Some(m.lower.1), Some(m.upper.1))
    };
    Memory {
        lower: (prev_l, l),
        upper: (prev_u, u),
    }
}

pub mod fractional;
pub mod integral;
