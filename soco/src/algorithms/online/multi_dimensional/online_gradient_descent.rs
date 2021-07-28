use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::norm::euclidean;
use crate::numerics::convex_optimization::{find_minimizer, WrappedObjective};
use crate::problem::{FractionalSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use finitediff::FiniteDiff;
use std::sync::Arc;

pub struct Options<'a> {
    /// Learning rates in each time step.
    pub eta: Arc<dyn Fn(i32) -> f64 + 'a>,
}

/// Online Gradient Descent
pub fn ogd(
    o: Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options,
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let t = xs.t_end();
    let default_x = Config::repeat(0., o.p.d);
    let x = if xs.is_empty() {
        default_x
    } else {
        let prev_x = xs.now();
        let f = |x: &Vec<f64>| o.p.hit_cost(t, Config::new(x.clone())).raw();
        let step =
            (options.eta)(t) * Config::new(prev_x.to_vec().central_diff(&f));
        project(o.p.bounds, prev_x - step)?
    };

    Ok(Step(x, None))
}

/// Projection of `y` under the Euclidean norm
fn project(
    bounds: Vec<(f64, f64)>,
    y: FractionalConfig,
) -> Result<FractionalConfig> {
    let objective = WrappedObjective::new(y, |x, y| {
        euclidean()(Config::new(x.to_vec()) - y.clone())
    });
    Ok(Config::new(find_minimizer(objective, bounds)?.0))
}
