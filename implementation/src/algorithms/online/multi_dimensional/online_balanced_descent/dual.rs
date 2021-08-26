use crate::algorithms::online::{FractionalStep, OnlineAlgorithm, Step};
use crate::config::Config;
use crate::distance::{
    dual_norm, euclidean, negative_entropy, norm_squared, DistanceGeneratingFn,
};
use crate::numerics::convex_optimization::find_minimizer_of_hitting_cost;
use crate::numerics::finite_differences::gradient;
use crate::numerics::roots::find_root;
use crate::problem::{FractionalSmoothedConvexOptimization, Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use crate::{
    algorithms::online::multi_dimensional::online_balanced_descent::meta::{
        obd, Options as MetaOptions,
    },
    model::{ModelOutputFailure, ModelOutputSuccess},
};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Balance parameter. $\eta > 0$.
    pub eta: f64,
    /// Distance-generating function.
    pub h: DistanceGeneratingFn<f64>,
}
impl Default for Options {
    fn default() -> Self {
        unimplemented!()
    }
}
#[pymethods]
impl Options {
    #[staticmethod]
    pub fn euclidean_squared(eta: f64) -> Self {
        Options {
            eta,
            h: norm_squared(euclidean()),
        }
    }

    #[staticmethod]
    pub fn negative_entropy(eta: f64) -> Self {
        Options {
            eta,
            h: negative_entropy(),
        }
    }
}

/// Dual Online Balanced Descent
pub fn dobd<C, D>(
    o: Online<FractionalSmoothedConvexOptimization<C, D>>,
    t: i32,
    xs: &FractionalSchedule,
    _: (),
    Options { eta, h }: Options,
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let prev_x = xs.now_with_default(Config::repeat(0., o.p.d));

    let (_, opt_) = find_minimizer_of_hitting_cost(
        t,
        o.p.hitting_cost.clone(),
        o.p.bounds.clone(),
    );
    let opt = opt_.raw();

    let a = opt;
    let b = o.p.hit_cost(t, prev_x.clone()).cost.raw();
    let l =
        find_root((a, if b.is_finite() { b } else { 1_000. * a }), |l: f64| {
            let Step(x, _) = obd
                .next(o.clone(), xs, None, MetaOptions { l, h: h.clone() })
                .unwrap();
            let f = |x: &Vec<f64>| {
                o.p.hit_cost(t, Config::new(x.clone())).cost.raw()
            };
            let h_ = |x: &Vec<f64>| h(Config::new(x.clone())).raw();
            let distance = dual_norm(o.p.switching_cost.clone())(
                Config::new(gradient(&h_, x.to_vec()))
                    - Config::new(gradient(&h_, prev_x.to_vec())),
            )
            .raw();
            let hitting_cost = dual_norm(o.p.switching_cost.clone())(
                Config::new(gradient(&f, x.to_vec())),
            )
            .raw();
            distance - eta * hitting_cost
        })
        .raw();

    obd.next(o, xs, None, MetaOptions { l, h })
}
