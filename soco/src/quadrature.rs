use crate::result::{Error, Result};
use crate::TOLERANCE;
use bacon_sci::integrate::{integrate, integrate_hermite, integrate_laguerre};

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
        return Err(Error::InvalidIntegrationInterval);
    };

    if result.is_nan() {
        Err(Error::Integration("Integration returned NaN".to_string()))
    } else {
        Ok(result)
    }
}

/// Uses the double exponential method (Tanh-sinh quadrature)
fn finite_integral(from: f64, to: f64, f: impl Fn(f64) -> f64) -> Result<f64> {
    integrate(from, to, f, TOLERANCE).map_err(Error::Integration)
}

/// Uses the Gaussian-Laguerre quadrature
fn semi_infinite_integral(f: impl Fn(f64) -> f64) -> Result<f64> {
    integrate_laguerre(|x| f(x) * std::f64::consts::E.powf(x), TOLERANCE)
        .map_err(Error::Integration)
}

/// Uses the Gaussian-Hermite quadrature
fn infinite_integral(f: impl Fn(f64) -> f64) -> Result<f64> {
    integrate_hermite(|x| f(x), TOLERANCE).map_err(Error::Integration)
}
