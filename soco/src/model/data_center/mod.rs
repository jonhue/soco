//! Utilities to model cost of data centers.

use noisy_float::prelude::*;

pub mod loads;
pub mod model;
pub mod models;

/// Ensures that `x` is greater than zero and handles edge cases appropriately.
pub fn safe_balancing(x: R64, total_load: R64, f: impl Fn() -> R64) -> R64 {
    if x > 0. {
        f()
    } else if total_load > 0. {
        r64(f64::INFINITY)
    } else {
        assert!(total_load == 0.);
        r64(0.)
    }
}
