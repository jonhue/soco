use super::DistanceGeneratingFn;
use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{
    find_unbounded_minimizer, WrappedObjective,
};
use crate::problem::{FractionalSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use finitediff::FiniteDiff;
use noisy_float::prelude::*;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Determines the l-level set used in each step by the algorithm.
    pub l: f64,
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
    fn constructor(l: f64, h: Py<PyAny>) -> Self {
        Options {
            l,
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

/// Online Balanced Descent (meta algorithm)
pub fn obd<C, D>(
    o: Online<FractionalSmoothedConvexOptimization<C, D>>,
    t: i32,
    xs: &FractionalSchedule,
    _: (),
    Options { l, h }: Options,
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let prev_x = xs.now_with_default(Config::repeat(0., o.p.d));
    let x = bregman_projection(h, l, o.p.hitting_cost, t, prev_x);
    Ok(Step(x, None))
}

#[derive(Clone)]
struct ObjectiveData {
    h: DistanceGeneratingFn,
    x: FractionalConfig,
}

#[derive(Clone)]
struct ConstraintData<'a, C, D> {
    f: CostFn<'a, FractionalConfig, C, D>,
    t: i32,
    l: f64,
}

/// Bregman projection of `x` onto a convex `l`-sublevel set `K` of `f`.
///
/// `h` must be `m`-strongly convex and `M`-Lipschitz smooth for the norm function with fixed `m` and `M`.
fn bregman_projection<C, D>(
    h: DistanceGeneratingFn,
    l: f64,
    f: CostFn<'_, FractionalConfig, C, D>,
    t: i32,
    x: FractionalConfig,
) -> FractionalConfig
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let d = x.d();
    let objective = WrappedObjective::new(ObjectiveData { h, x }, |y, data| {
        bregman_divergence(&data.h, Config::new(y.to_vec()), data.x.clone())
    });
    // `l`-sublevel set of `f`
    let constraint =
        WrappedObjective::new(ConstraintData { f, t, l }, |y, data| {
            data.f.call_certain(data.t, Config::new(y.to_vec())).cost
                - n64(data.l)
        });

    let (y, _) = find_unbounded_minimizer(objective, d, vec![constraint]);
    Config::new(y)
}

/// Bregman divergence between `x` and `y`.
fn bregman_divergence(
    h: &DistanceGeneratingFn,
    x: FractionalConfig,
    y: FractionalConfig,
) -> N64 {
    let m = |x: &Vec<f64>| h(Config::new(x.clone()));
    let mx = h(x.clone());
    let my = h(y.clone());
    let grad = Config::new(y.to_vec().central_diff(&m));
    n64(mx - my - grad * (x - y))
}
