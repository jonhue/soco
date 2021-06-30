//! Utilities modeling the cost of server load.
//!
//! This module contains a number of models with varying complexity and the
//! building blocks they are made up of.

use crate::cost::data_center::{load_balance, LoadCostFn};
use crate::utils::pos;
use crate::value::Value;
use num::ToPrimitive;
use std::sync::Arc;

/// Multiplicative energy loss. Returns the power consumed by a server as a
/// function of load `l` according to the formula `l^a + b` where `l^a` models
/// the dynamic power while `b` models the static/leakage power.
///
/// * `a > 1`
/// * `b >= 0`
pub fn energy_loss_mul(a: f64, b: f64, l: f64) -> f64 {
    l.powf(a) + b
}

/// Additive energy loss. Returns the power consumed by a server as a function
/// of load `l` according to the formula `e_0 + e_1 * l` where `e_1 * l` models
/// the dynamic power while `e_0` models the static/leakage power.
pub fn energy_loss_add(e_0: f64, e_1: f64, l: f64) -> f64 {
    e_0 + e_1 * l
}

/// Returns the revenue loss given average delay `d` and load `l`
/// according to the formula `d_1 * l * (d - d_0)^+` where `d_0` is the minimum
/// delay users can detect and `d_1` is a constant.
pub fn revenue_loss(d_0: f64, d_1: f64, d: f64, l: f64) -> f64 {
    d_1 * l * pos(d - d_0)
}

/// Returns the average delay of a server modeled by an M/GI/1 Processor Sharing queue
/// where the service rate of the server is `mu`.
pub fn queueing_delay(mu: f64, l: f64) -> f64 {
    1. / (mu - l)
}

/// A simple model that only takes into account energy consumption (modeled multiplicatively).
pub fn m1<'a, T>(a: &'a Vec<f64>, b: &'a Vec<f64>) -> LoadCostFn<'a, T>
where
    T: Value + 'a,
{
    load_balance(Arc::new(move |_, k, l| {
        energy_loss_mul(a[k], b[k], l.total())
    }))
}

/// A model that takes into account the energy consumption (modeled additively) and
/// the revenue loss based on the average delay in a M/GI/1 Processor Sharing
/// queue with service rate `service_rate`.
pub fn m2<'a, T>(
    e_0: &'a Vec<f64>,
    e_1: &'a Vec<f64>,
    d_0: &'a Vec<f64>,
    d_1: &'a Vec<f64>,
    service_rate: &'a Vec<f64>,
) -> LoadCostFn<'a, T>
where
    T: Value + 'a,
{
    load_balance(Arc::new(move |_, k, l| {
        let sum_l = l.total();
        let d = queueing_delay(service_rate[k], sum_l);
        let r = revenue_loss(d_0[k], d_1[k], d, sum_l);
        let e = energy_loss_add(e_0[k], e_1[k], sum_l);
        r + e
    }))
}

/// A model that takes into account the energy consumption (modeled additively)
/// with some free energy for each server type (i.e. data center) and
/// the revenue loss based on the average delay in a M/GI/1 Processor Sharing
/// queue with service rate `service_rate` and some network delay between load
/// type and data center.
///
/// * `p` - cost of electricity per server of type `k`
/// * `r` - number of servers of type `k` that can be powered by "free" energy generated on-site at time `t`
/// * `network_delay` - network delay at time `t` for load type `j` and server type `k`
/// * `service_rate` - service rate of processor queue
/// * `g` - revenue loss by delay
pub fn m3<'a, T>(
    p: &'a Vec<f64>,
    r: Arc<dyn Fn(i32, usize) -> f64 + 'a>,
    network_delay: Arc<dyn Fn(i32, usize, usize) -> f64 + 'a>,
    service_rate: &'a Vec<f64>,
    g: f64,
) -> LoadCostFn<'a, T>
where
    T: Value + 'a,
{
    Arc::new(move |t, k, l| {
        let r = r.clone();
        let network_delay = network_delay.clone();
        Arc::new(move |x| {
            let e = p[k] * pos(ToPrimitive::to_f64(&x).unwrap() - r(t, k));

            let mut d = 0.;
            for j in 0..l.e() as usize {
                d += g
                    * l[j]
                    * (queueing_delay(
                        service_rate[k],
                        l.total() / ToPrimitive::to_f64(&x).unwrap(),
                    ) + network_delay(t, j, k));
            }

            d + e
        })
    })
}
