//! Differentiation.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use bacon_sci::differentiate::{
    derivative as derivative_, second_derivative as second_derivative_,
};
use noisy_float::prelude::*;

pub fn derivative(f: impl Fn(f64) -> f64, x: f64) -> N64 {
    n64(derivative_(f, x, TOLERANCE).apply_precision())
}

pub fn second_derivative(f: impl Fn(f64) -> f64, x: f64) -> N64 {
    n64(second_derivative_(f, x, TOLERANCE.powf(-0.25)).apply_precision())
}
