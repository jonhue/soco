use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::distance::{
    euclidean, negative_entropy, norm_squared, DistanceGeneratingFn,
};
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{
    find_minimizer_of_hitting_cost, minimize, WrappedObjective,
};
use crate::numerics::finite_differences::gradient;
use crate::problem::{FractionalSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use noisy_float::prelude::*;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Determines the l-level set used in each step by the algorithm.
    pub l: f64,
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
    pub fn euclidean_squared(l: f64) -> Self {
        Options {
            l,
            h: norm_squared(euclidean()),
        }
    }

    #[staticmethod]
    pub fn negative_entropy(l: f64) -> Self {
        Options {
            l,
            h: negative_entropy(),
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
    assert!(l.is_finite());

    let prev_x = xs.now_with_default(Config::repeat(0., o.p.d));
    let x =
        bregman_projection(h, n64(l), o.p.bounds, o.p.hitting_cost, t, prev_x);
    Ok(Step(x, None))
}

#[derive(Clone)]
struct ObjectiveData {
    h: DistanceGeneratingFn<f64>,
    x: FractionalConfig,
}

#[derive(Clone)]
struct ConstraintData<'a, C, D> {
    f: CostFn<'a, FractionalConfig, C, D>,
    t: i32,
    l: N64,
}

/// Bregman projection of `x` onto a convex `l`-sublevel set `K` of `f`.
///
/// `h` must be `m`-strongly convex and `M`-Lipschitz smooth for the norm function with fixed `m` and `M`.
fn bregman_projection<C, D>(
    h: DistanceGeneratingFn<f64>,
    l: N64,
    bounds: Vec<(f64, f64)>,
    f: CostFn<'_, FractionalConfig, C, D>,
    t: i32,
    x: FractionalConfig,
) -> FractionalConfig
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let (v, _) = find_minimizer_of_hitting_cost(t, f.clone(), bounds.clone());

    let objective = WrappedObjective::new(ObjectiveData { h, x }, |y, data| {
        bregman_divergence(&data.h, Config::new(y.to_vec()), data.x.clone())
    });
    // `l`-sublevel set of `f`
    let constraint = WrappedObjective::new(
        ConstraintData { f: f.clone(), t, l },
        |y, data| {
            let v = data.f.call_certain(data.t, Config::new(y.to_vec())).cost;
            v - data.l
        },
    );

    let (y, _) = minimize(objective, bounds, Some(v.clone()), vec![constraint]);

    if f.call_certain(t, Config::new(y.clone())).cost > l {
        // distance minimization failed
        Config::new(v)
    } else {
        Config::new(y)
    }
}

/// Bregman divergence between `x` and `y`.
fn bregman_divergence(
    h: &DistanceGeneratingFn<f64>,
    x: FractionalConfig,
    y: FractionalConfig,
) -> N64 {
    let m = |x: &Vec<f64>| h(Config::new(x.clone())).raw();
    let mx = h(x.clone()).raw();
    let my = h(y.clone()).raw();
    let grad = Config::new(gradient(&m, y.to_vec()));
    let d = mx - my - grad * (x - y);
    n64(d)
}
