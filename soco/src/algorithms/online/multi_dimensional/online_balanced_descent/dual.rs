use super::DistanceGeneratingFn;
use crate::algorithms::online::{FractionalStep, OnlineAlgorithm, Step};
use crate::config::{Config, FractionalConfig};
use crate::norm::dual;
use crate::numerics::convex_optimization::find_minimizer_of_hitting_cost;
use crate::numerics::roots::find_root;
use crate::problem::{FractionalSmoothedConvexOptimization, Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use crate::{
    algorithms::online::multi_dimensional::online_balanced_descent::{
        meta::{obd, Options as MetaOptions},
        MAX_L_FACTOR,
    },
    model::{ModelOutputFailure, ModelOutputSuccess},
};
use finitediff::FiniteDiff;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Balance parameter. `eta > 0`.
    pub eta: f64,
    /// Distance-generating function.
    pub h: DistanceGeneratingFn,
}
impl Default for Options {
    fn default() -> Self {
        unimplemented!()
    }
}
#[pymethods]
impl Options {
    #[new]
    fn constructor(eta: f64, h: Py<PyAny>) -> Self {
        Options {
            eta,
            h: Arc::new(move |x| {
                Python::with_gil(|py| {
                    h.call1(py, (x,))
                        .expect("options `h` method invalid")
                        .extract(py)
                        .expect("options `h` method invalid")
                })
            }),
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

    let v = Config::new(
        find_minimizer_of_hitting_cost(
            t,
            o.p.hitting_cost.clone(),
            o.p.bounds.clone(),
        )
        .0,
    );
    let minimal_hitting_cost = o.p.hit_cost(t, v).cost.raw();

    let a = minimal_hitting_cost;
    let b = MAX_L_FACTOR * minimal_hitting_cost;
    let l = find_root((a, b), |l: f64| {
        let mut xs = xs.clone(); // remove this!
        balance_function(&o, &mut xs, &prev_x, t, l, eta, h.clone())
    })
    .raw();

    obd.next(o, xs, None, MetaOptions { l, h })
}

fn balance_function<C, D>(
    o: &Online<FractionalSmoothedConvexOptimization<C, D>>,
    xs: &mut FractionalSchedule,
    prev_x: &FractionalConfig,
    t: i32,
    l: f64,
    eta: f64,
    h: DistanceGeneratingFn,
) -> f64
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let Step(x, _) = obd
        .next(o.clone(), xs, None, MetaOptions { l, h: h.clone() })
        .unwrap();
    let f = |x: &Vec<f64>| o.p.hit_cost(t, Config::new(x.clone())).cost.raw();
    let m = |x: &Vec<f64>| h(Config::new(x.clone()));
    let distance = dual(&o.p.switching_cost)(
        Config::new(x.to_vec().central_diff(&m))
            - Config::new(prev_x.to_vec().central_diff(&m)),
    )
    .raw();
    let hitting_cost =
        dual(&o.p.switching_cost)(Config::new(x.to_vec().central_diff(&f)))
            .raw();
    distance / hitting_cost - eta
}
