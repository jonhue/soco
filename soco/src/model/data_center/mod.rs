//! Utilities to model cost of data centers.

use noisy_float::prelude::*;

pub mod loads;
pub mod model;
pub mod models;

/// Ensures that `x` is greater than zero and handles edge cases appropriately.
pub fn safe_balancing(x: N64, total_load: N64, f: impl Fn() -> N64) -> N64 {
    if x > 0. {
        f()
    } else if total_load > 0. {
        n64(f64::INFINITY)
    } else {
        assert!(total_load == 0.);
        n64(0.)
    }
}
