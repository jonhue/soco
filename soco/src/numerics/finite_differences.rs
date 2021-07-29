//! Differentiation.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use bacon_sci::differentiate::{
    derivative as derivative_, second_derivative as second_derivative_,
};
use log::debug;
use noisy_float::prelude::*;

pub fn derivative(f: impl Fn(f64) -> f64, x: f64) -> N64 {
    let y = f(x);
    if y.is_infinite() {
        return n64(f64::INFINITY);
    }
    let r = derivative_(f, x, TOLERANCE).apply_precision();
    if r.is_nan() {
        debug!("x={};f(x)={}", x, y);
        return n64(0.);
    }
    n64(r)
}

pub fn second_derivative(f: impl Fn(f64) -> f64, x: f64) -> N64 {
    let y = f(x);
    if y.is_infinite() {
        return n64(f64::INFINITY);
    }
    let r = second_derivative_(f, x, TOLERANCE.powf(-0.25)).apply_precision();
    if r.is_nan() {
        debug!("x={};f(x)={}", x, y);
        return n64(0.);
    }
    n64(r)
}
