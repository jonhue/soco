use rand::prelude::*;
use rand_pcg::Pcg64;
use soco::config::{FractionalConfig, IntegralConfig};

/// Returns `t` if all dimensions are `0`, `0` otherwise.
pub fn penalize_zero(t: i32, j: IntegralConfig) -> f64 {
    t as f64
        * if (0..j.d() as usize).all(|k| j[k] <= 0) {
            1.
        } else {
            0.
        }
}

/// Returns random costs (which must not necessarily be convex).
pub fn random(t: i32, j: IntegralConfig) -> f64 {
    let r: f64 = j
        .to_vec()
        .into_iter()
        .enumerate()
        .map(|(k, i)| {
            Pcg64::seed_from_u64(t as u64 * k as u64).gen_range(0.0..1.)
                * i as f64
        })
        .sum();
    Pcg64::seed_from_u64(t as u64 * r as u64).gen_range(0.0..1_000_000.)
}

/// `t * exp(-x)` for a single dimension.
pub fn inv_e(t: i32, j: FractionalConfig) -> f64 {
    assert!(j.d() == 1);
    t as f64 * std::f64::consts::E.powf(-j[0])
}

/// `1 / t * x^2` for a single dimension.
pub fn parabola(t: i32, j: FractionalConfig) -> f64 {
    assert!(j.d() == 1);
    1. / t as f64 * (j[0] as f64).powi(2)
}

/// `1 / t * x^2` for a single dimension.
pub fn int_parabola(t: i32, j: IntegralConfig) -> f64 {
    assert!(j.d() == 1);
    1. / t as f64 * (j[0] as f64).powi(2)
}
