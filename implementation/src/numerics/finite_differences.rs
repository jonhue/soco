//! Differentiation.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use bacon_sci_1::differentiate::{
    derivative as derivative_, second_derivative as second_derivative_,
};
use finitediff::FiniteDiff;
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

pub fn gradient(f: &impl Fn(&Vec<f64>) -> f64, xs: Vec<f64>) -> Vec<f64> {
    let result = xs.central_diff(f);
    result.iter().map(|&d| {
        if d.is_nan() {
            warn!(
                "First-order finite difference returned NaN. Assuming result `0`."
            );
            return 0.;
        }
        d
    }).collect()
}
