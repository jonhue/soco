#![allow(clippy::float_cmp)]

use std::sync::Arc;

use crate::algorithms::optimization::find_minimizer;
use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;

pub struct Options {
    /// Convexity parameter. Chosen such that `f_t(x) \geq f_t(v_t) + \frac{m}{2} \norm{x - v_t}_2^2` where `v_t` is the minimizer of `f_t`.
    pub m: f64,
    /// Convexity parameter of potential function of Bregman convergence.
    pub alpha: f64,
    /// Smoothness parameter of potential function of Bregman convergence.
    pub beta: f64,
}

/// Regularized Online Balanced Descent
pub fn robd(
    o: &Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options,
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;

    let (lambda_1, lambda_2) =
        build_parameters(options.m, options.alpha, options.beta);

    let t = xs.t_end() + 1;
    let prev_x = if xs.is_empty() {
        Config::repeat(0., o.p.d)
    } else {
        xs.now().clone()
    };

    let v = find_minimizer(t, &o.p.hitting_cost, &o.p.bounds)?;
    let regularization_function: CostFn<'_, FractionalConfig> =
        Arc::new(|t, x| {
            Some(
                (o.p.hitting_cost)(t, x.clone())?
                    + lambda_1
                        * (o.p.switching_cost)(x.clone() - prev_x.clone())
                    + lambda_2 * (o.p.switching_cost)(x - v.clone()),
            )
        });
    let x = find_minimizer(t, &regularization_function, &o.p.bounds)?;
    Ok(Step(x, None))
}

/// Determines `lambda_1` (weight of movement cost) and `lambda_2` (weight of regularizer).
fn build_parameters(m: f64, alpha: f64, beta: f64) -> (f64, f64) {
    let f_lambda_2 = |lambda_1| {
        (lambda_1 * m / 2.
            * (1. + (1. + 4. * beta.powi(2) / (alpha * m)).sqrt())
            - m)
            / beta
    };

    let mut lambda_2 = 0.;
    let mut lambda_1 =
        2. / (1. + (1. + 4. * beta.powi(2) / (alpha * m)).sqrt());
    if f_lambda_2(lambda_1) == lambda_2 {
        return (lambda_1, lambda_2);
    }

    lambda_1 = 1.;
    lambda_2 = f_lambda_2(lambda_1);
    (lambda_1, lambda_2)
}