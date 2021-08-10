//! Root finding.

use super::PRECISION;
use crate::numerics::{ApplicablePrecision, TOLERANCE};
use bacon_sci::roots::brent;
use log::warn;
use noisy_float::prelude::*;

// in practice, the inputs to the function may be imprecise enough to lead to the error:
// `initial guesses do not bracket root` if the actual precision is required as the
// function value at the boundary may deviate slightly from `0`.
static ROOT_PRECISION: f64 = PRECISION * 1_000.;

pub fn find_root(interval: (f64, f64), f: impl Fn(f64) -> f64) -> N64 {
    if (interval.0 - interval.1).abs() < PRECISION {
        return n64(interval.0);
    }

    let l = f(interval.0);
    let r = f(interval.1);
    if l.abs() < ROOT_PRECISION {
        n64(interval.0)
    } else if r.abs() < ROOT_PRECISION {
        n64(interval.1)
    } else if l * r > 0. {
        warn!("Interval does not contain root. This may be the result of an earlier numerical inaccuracy. Assuming the boundary with function value closest to `0` is the root.");
        if l.abs() <= r.abs() {
            n64(interval.0)
        } else {
            n64(interval.1)
        }
    } else {
        n64(brent(
            interval,
            |x| if x.is_nan() { f64::NAN } else { f(x) },
            TOLERANCE,
        )
        .unwrap()
        .apply_precision())
    }
}
