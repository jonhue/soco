use noisy_float::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;
use soco::{
    config::{FractionalConfig, IntegralConfig},
    cost::{FailableCost, FailableCostFn, RawCostFn, SingleCostFn},
    model::{data_center::DataCenterModelOutputFailure, ModelOutputFailure},
    vec_wrapper::VecWrapper,
};

fn wrap<'a, T, D>(
    f: impl Fn(i32, T) -> f64 + Send + Sync + 'a,
) -> FailableCostFn<'a, T, D>
where
    T: Clone,
    D: ModelOutputFailure,
{
    FailableCostFn::new(
        1,
        SingleCostFn::certain(move |t, x| FailableCost::raw(n64(f(t, x)))),
    )
}

/// Returns $1$.
pub fn constant() -> RawCostFn<'static, IntegralConfig> {
    wrap(|_, _| 1.)
}

/// Returns $1$.
pub fn constant_simple(
) -> FailableCostFn<'static, f64, DataCenterModelOutputFailure> {
    wrap(|_, _| 1.)
}

/// Returns $t$ if all dimensions are $0$, $0$ otherwise.
pub fn penalize_zero() -> RawCostFn<'static, IntegralConfig> {
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
pub fn random() -> RawCostFn<'static, IntegralConfig> {
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

/// $t \cdot exp(-x)$ for multiple dimensions.
pub fn inv_e() -> RawCostFn<'static, FractionalConfig> {
    wrap(|t: i32, j: FractionalConfig| {
        t as f64 * j.iter().map(|&x| std::f64::consts::E.powf(-x)).sum::<f64>()
    })
}

/// $t \cdot exp(-x)$ for multiple dimensions.
pub fn inv_e_sblo() -> FailableCostFn<'static, f64, DataCenterModelOutputFailure>
{
    wrap(|t: i32, x: f64| t as f64 * std::f64::consts::E.powf(-x))
}

/// $1 / t \cdot (x-1)^2$ around $1$ for a single dimension.
pub fn parabola() -> RawCostFn<'static, FractionalConfig> {
    wrap(|t: i32, j: FractionalConfig| {
        assert!(j.d() == 1);
        1. / t as f64 * (j[0] as f64 - 1.).powi(2)
    })
}

/// $1 / t \cdot (x-1)^2$ around $1$ for a single dimension.
pub fn int_parabola() -> RawCostFn<'static, IntegralConfig> {
    wrap(|t: i32, j: IntegralConfig| {
        assert!(j.d() == 1);
        1. / t as f64 * (j[0] as f64 - 1.).powi(2)
    })
}
