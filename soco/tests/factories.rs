use noisy_float::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;
use soco::{
    config::{FractionalConfig, IntegralConfig},
    cost::{CostFn, SingleCostFn},
    vec_wrapper::VecWrapper,
};

fn wrap<'a, T>(f: impl Fn(i32, T) -> f64 + Send + Sync + 'a) -> CostFn<'a, T>
where
    T: Clone,
{
    CostFn::new(1, SingleCostFn::certain(move |t, x| n64(f(t, x))))
}

/// Returns `1`.
pub fn constant() -> CostFn<'static, IntegralConfig> {
    wrap(|_, _| 1.)
}

/// Returns `t` if all dimensions are `0`, `0` otherwise.
pub fn penalize_zero() -> CostFn<'static, IntegralConfig> {
    wrap(|t: i32, j: IntegralConfig| {
        t as f64
            * if (0..j.d() as usize).all(|k| j[k] <= 0) {
                1.
            } else {
                0.
            }
    })
}

/// Returns random costs (which must not necessarily be convex).
pub fn random() -> CostFn<'static, IntegralConfig> {
    wrap(|t: i32, j: IntegralConfig| {
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
    })
}

/// `t * exp(-x)` for multiple dimensions.
pub fn inv_e() -> CostFn<'static, FractionalConfig> {
    wrap(|t: i32, j: FractionalConfig| {
        t as f64 * j.iter().map(|x| std::f64::consts::E.powf(-x)).sum::<f64>()
    })
}

/// `1 / t * x^2` for a single dimension.
pub fn parabola() -> CostFn<'static, FractionalConfig> {
    wrap(|t: i32, j: FractionalConfig| {
        assert!(j.d() == 1);
        1. / t as f64 * (j[0] as f64).powi(2)
    })
}

/// `1 / t * x^2` for a single dimension.
pub fn int_parabola() -> CostFn<'static, IntegralConfig> {
    wrap(|t: i32, j: IntegralConfig| {
        assert!(j.d() == 1);
        1. / t as f64 * (j[0] as f64).powi(2)
    })
}
