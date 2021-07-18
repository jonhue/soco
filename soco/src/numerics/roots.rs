//! Root finding.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use crate::result::{Failure, Result};
use bacon_sci::roots::brent;
use noisy_float::prelude::*;

pub fn find_root(
    interval: (f64, f64),
    f: impl FnMut(f64) -> f64,
) -> Result<R64> {
    Ok(r64(brent(interval, f, TOLERANCE)
        .map_err(Failure::Bisection)?
        .apply_precision()))
}
