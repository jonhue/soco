use crate::numerics::{ApplicablePrecision, TOLERANCE};
use crate::result::{Failure, Result};
use bacon_sci::integrate::{integrate, integrate_hermite, integrate_laguerre};

pub mod piecewise;

/// Integrates `f` from `from` to `to` using an applicable quadrature method.
pub fn integral(from: f64, to: f64, f: impl Fn(f64) -> f64) -> Result<f64> {
    let result = if from == f64::NEG_INFINITY && to == f64::INFINITY {
        infinite_integral(f)?
    } else if to == f64::INFINITY {
        semi_infinite_integral(|x| f(from + x))?
    } else if from == f64::NEG_INFINITY {
        semi_infinite_integral(|x| f(to - x))?
    } else if from != f64::NEG_INFINITY && to != f64::INFINITY {
        finite_integral(from, to, f)?
    } else {
        return Err(Failure::InvalidInterval { from, to });
    };

    if result.is_nan() {
        Err(Failure::Integration("returned NaN".to_string()))
    } else {
        Ok(result.apply_precision())
    }
}

/// Uses the double exponential method (Tanh-sinh quadrature)
fn finite_integral(from: f64, to: f64, f: impl Fn(f64) -> f64) -> Result<f64> {
    integrate(from, to, f, TOLERANCE).map_err(Failure::Integration)
}

/// Uses the Gaussian-Laguerre quadrature
fn semi_infinite_integral(f: impl Fn(f64) -> f64) -> Result<f64> {
    integrate_laguerre(|x| f(x) * std::f64::consts::E.powf(x), TOLERANCE)
        .map_err(Failure::Integration)
}

/// Uses the Gaussian-Hermite quadrature
fn infinite_integral(f: impl Fn(f64) -> f64) -> Result<f64> {
    integrate_hermite(|x| f(x), TOLERANCE).map_err(Failure::Integration)
}
