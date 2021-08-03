use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::norm::euclidean;
use crate::numerics::convex_optimization::{find_minimizer, WrappedObjective};
use crate::problem::{FractionalSmoothedConvexOptimization, Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use finitediff::FiniteDiff;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Learning rates in each time step.
    pub eta: Arc<dyn Fn(i32) -> f64 + Send + Sync>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            eta: Arc::new(|t| 1. / (t as f64).sqrt()),
        }
    }
}
#[pymethods]
impl Options {
    #[new]
    fn constructor(eta: Py<PyAny>) -> Self {
        Options {
            eta: Arc::new(move |t| {
                Python::with_gil(|py| {
                    eta.call1(py, (t,))
                        .expect("options `eta` method invalid")
                        .extract(py)
                        .expect("options `eta` method invalid")
                })
            }),
        }
    }
}

/// Online Gradient Descent
pub fn ogd<C, D>(
    o: Online<FractionalSmoothedConvexOptimization<C, D>>,
    mut t: i32,
    xs: &FractionalSchedule,
    _: (),
    options: Options,
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    // apply lookahead
    t += 1;

    let prev_x = xs.now_with_default(Config::repeat(0., o.p.d));
    let f =
        |x: &Vec<f64>| o.p.hit_cost(t - 1, Config::new(x.clone())).cost.raw();
    let step =
        (options.eta)(t - 1) * Config::new(prev_x.to_vec().central_diff(&f));
    let x = project(o.p.bounds, prev_x - step);

    Ok(Step(x, None))
}

/// Projection of `y` under the Euclidean norm
fn project(bounds: Vec<(f64, f64)>, y: FractionalConfig) -> FractionalConfig {
    let objective = WrappedObjective::new(y, |x, y| {
        euclidean()(Config::new(x.to_vec()) - y.clone())
    });
    Config::new(find_minimizer(objective, bounds).0)
}
