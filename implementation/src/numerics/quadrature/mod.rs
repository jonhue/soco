//! Integration.

use crate::numerics::{ApplicablePrecision, TOLERANCE};
use bacon_sci::integrate::{integrate, integrate_hermite, integrate_laguerre};
use log::debug;
use noisy_float::prelude::*;

pub mod piecewise;

/// Integrates $f$ from $from$ to $to$ using an applicable quadrature method.
pub fn integral(from: f64, to: f64, f: impl Fn(f64) -> f64) -> N64 {
    let result = if from == f64::NEG_INFINITY && to == f64::INFINITY {
        infinite_integral(f)
    } else if to == f64::INFINITY {
        semi_infinite_integral(|x| f(from + x))
    } else if from == f64::NEG_INFINITY {
        semi_infinite_integral(|x| f(to - x))
    } else if from != f64::NEG_INFINITY && to != f64::INFINITY {
        finite_integral(from, to, f)
    } else {
        panic!("The interval from {} to {} is invalid.", from, to)
    };

    if result.is_nan() {
        panic!("Integration from {} to {} returned NaN.", from, to)
    } else {
        n64(result.raw().apply_precision())
    }
}

/// Uses the double exponential method (Tanh-sinh quadrature)
fn finite_integral(from: f64, to: f64, f: impl Fn(f64) -> f64) -> N64 {
    handle_integration_result(integrate(from, to, f, TOLERANCE))
}

/// Uses the Gaussian-Laguerre quadrature
fn semi_infinite_integral(f: impl Fn(f64) -> f64) -> N64 {
    n64(
        integrate_laguerre(|x| f(x) * std::f64::consts::E.powf(x), TOLERANCE)
            .unwrap(),
    )
}

/// Uses the Gaussian-Hermite quadrature
fn infinite_integral(f: impl Fn(f64) -> f64) -> N64 {
    n64(integrate_hermite(|x| f(x), TOLERANCE).unwrap())
}

fn handle_integration_result(
    result: std::result::Result<f64, (f64, String)>,
) -> N64 {
    n64(match result {
        Ok(integral) => integral,
        Err((integral, error)) => {
            debug!("Integration failed with message: {}", error);
            integral
        }
    })
}
