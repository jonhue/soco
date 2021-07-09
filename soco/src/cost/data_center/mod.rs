//! Utilities to model cost of data centers.

pub mod loads;
pub mod model;
pub mod models;

/// Ensures that `x` is greater than zero and handles edge cases appropriately.
pub fn safe_balancing(x: f64, total_load: f64, f: impl Fn() -> f64) -> f64 {
    if x > 0. {
        f()
    } else if total_load > 0. {
        f64::INFINITY
    } else {
        assert!(total_load == 0.);
        0.
    }
}
