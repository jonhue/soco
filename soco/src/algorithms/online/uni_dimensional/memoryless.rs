use crate::algorithms::online::{FractionalStep, Step};
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{minimize, WrappedObjective};
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization, Online, Problem,
};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use log::debug;
use noisy_float::prelude::*;

/// Memoryless Algorithm. Special case of Primal Online Balanced Descent.
pub fn memoryless<C, D>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>>,
    t: i32,
    xs: &FractionalSchedule,
    _: (),
    _: (),
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;
    assert(o.p.d == 1, Failure::UnsupportedProblemDimension(o.p.d))?;

    let prev_x = xs.now_with_default(Config::single(0.))[0];

    let x = next(o, t, prev_x)?;
    debug!("determined next config: {:?}", x);
    Ok(Step(Config::single(x), None))
}

#[derive(Clone)]
struct ObjectiveData<'a, C, D> {
    t: i32,
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>>,
}

/// Determines next `x` with a convex optimization.
fn next<C, D>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>>,
    t: i32,
    prev_x: f64,
) -> Result<f64>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let bounds = vec![(0., o.p.bounds[0])];
    let data = ObjectiveData { t, o };
    let objective = WrappedObjective::new(data.clone(), |xs, data| {
        data.o.p.hit_cost(data.t, Config::new(xs.to_vec())).cost
    });
    let constraint = WrappedObjective::new(data, |xs, data| {
        n64(data.o.p.switching_cost[0]) * n64((xs[0] - prev_x).abs())
            - data.o.p.hit_cost(data.t, Config::single(xs[0])).cost / n64(2.)
    });

    let (xs, _) = minimize(objective, bounds, None, vec![constraint])?;
    Ok(xs[0])
}
