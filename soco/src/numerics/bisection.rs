//! Root finding.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use crate::result::{Failure, Result};
use bacon_sci::roots::bisection as bisection_;

/// Maximum number of bisection iterations.
static MAX_ITERATIONS: usize = 1_000;

pub fn bisection(
    interval: (f64, f64),
    f: impl FnMut(f64) -> f64,
) -> Result<f64> {
    Ok(bisection_(interval, f, TOLERANCE, MAX_ITERATIONS)
        .map_err(Failure::Bisection)?
        .apply_precision())
}
