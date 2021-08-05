use super::DistanceGeneratingFn;
use crate::algorithms::online::{FractionalStep, OnlineAlgorithm, Step};
use crate::config::Config;
use crate::numerics::convex_optimization::find_minimizer_of_hitting_cost;
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
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// The movement cost is at most `beta` times the hitting cost. `beta > 0`.
    pub beta: f64,
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
    fn constructor(beta: f64, h: Py<PyAny>) -> Self {
        Options {
            beta,
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

/// Primal Online Balanced Descent
pub fn pobd<C, D>(
    o: Online<FractionalSmoothedConvexOptimization<C, D>>,
    t: i32,
    xs: &FractionalSchedule,
    _: (),
    Options { beta, h }: Options,
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let prev_x = xs.now_with_default(Config::repeat(0., o.p.d));

    let (v_, opt_) = find_minimizer_of_hitting_cost(
        t,
        o.p.hitting_cost.clone(),
        o.p.bounds.clone(),
    );
    let v = Config::new(v_);
    let opt = opt_.raw();

    if (o.p.switching_cost)(v.clone() - prev_x.clone()).raw() <= beta * opt {
        return Ok(Step(v, None));
    }

    let a = opt;
    let b = o.p.hit_cost(t, prev_x.clone()).cost.raw();
    let l = find_root((a, b), |l: f64| {
        let Step(x, _) = obd
            .next(o.clone(), xs, None, MetaOptions { l, h: h.clone() })
            .unwrap();
        (o.p.switching_cost)(x - prev_x.clone()).raw() - beta * l
    })
    .raw();

    obd.next(o, xs, None, MetaOptions { l, h })
}
