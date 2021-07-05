//! Root finding.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use crate::result::{Failure, Result};
use bacon_sci::roots::brent;

pub fn find_root(
    interval: (f64, f64),
    f: impl FnMut(f64) -> f64,
) -> Result<f64> {
    Ok(brent(interval, f, TOLERANCE)
        .map_err(Failure::Bisection)?
        .apply_precision())
}
