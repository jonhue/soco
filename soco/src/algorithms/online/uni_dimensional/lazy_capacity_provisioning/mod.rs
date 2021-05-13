//! Lazy Capacity Provisioning.

/// Lower and upper bound at some time `t`.
pub struct Memory<T>(pub T, pub T);

/// Finds a valid reference time to base the optimization on (alternatively to time `0`).
pub fn find_initial_time<T: PartialOrd>(
    final_t: i32,
    ms: &Vec<Memory<T>>,
) -> i32 {
    for t in (2..final_t).rev() {
        if is_valid_initial_time(&ms[t as usize - 2], &ms[t as usize - 1]) {
            return t;
        }
    }

    0
}

/// Returns `true` if the time `t` with current bounds `m` and previous bounds (at `t - 1`) `prev_m` may be used as reference time.
fn is_valid_initial_time<T: PartialOrd>(
    m: &Memory<T>,
    prev_m: &Memory<T>,
) -> bool {
    m.1 < prev_m.1 || m.0 > prev_m.0
}

pub mod fractional;
pub mod integral;
