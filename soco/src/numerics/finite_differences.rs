//! Differentiation.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use bacon_sci::differentiate::{
    derivative as derivative_, second_derivative as second_derivative_,
};
use log::warn;
use noisy_float::prelude::*;

pub fn derivative(f: impl Fn(f64) -> f64, x: f64) -> N64 {
    if f(x).is_infinite() {
        return n64(f64::INFINITY);
    }

    let result = derivative_(f, x, TOLERANCE).apply_precision();
    if result.is_nan() {
        warn!(
            "First-order finite difference returned NaN. Assuming result `0`."
        );
        return n64(0.);
    }
    n64(result)
}

pub fn second_derivative(f: impl Fn(f64) -> f64, x: f64) -> N64 {
    if f(x).is_infinite() {
        return n64(f64::INFINITY);
    }
    let result =
        second_derivative_(f, x, TOLERANCE.powf(-0.25)).apply_precision();
    if result.is_nan() {
        warn!(
            "Second-order finite difference returned NaN. Assuming result `0`."
        );
        return n64(0.);
    }
    n64(result)
}
