use crate::algorithms::online::{FractionalStep, Step};
use crate::config::Config;
use crate::numerics::convex_optimization::minimize;
use crate::problem::{FractionalSimplifiedSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use noisy_float::prelude::*;
use std::sync::Arc;

/// Memoryless Algorithm. Special case of Primal Online Balanced Descent.
pub fn memoryless(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    xs: &FractionalSchedule,
    _: (),
    _: (),
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;
    assert(o.p.d == 1, Failure::UnsupportedProblemDimension(o.p.d))?;

    let prev_x = xs.now_with_default(Config::single(0.))[0];

    let x = next(o, t, prev_x)?;
    Ok(Step(Config::single(x), None))
}

/// Determines next `x` with a convex optimization.
fn next(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    prev_x: f64,
) -> Result<f64> {
    let bounds = vec![(0., o.p.bounds[0])];
    let objective =
        |xs: &[f64]| -> N64 { o.p.hit_cost(t, Config::new(xs.to_vec())) };
    let constraint = Arc::new(|xs: &[f64]| -> N64 {
        n64(o.p.switching_cost[0]) * n64((xs[0] - prev_x).abs())
            - o.p.hit_cost(t, Config::single(xs[0])) / n64(2.)
    });

    let (xs, _) = minimize(objective, &bounds, None, vec![constraint], vec![])?;
    Ok(xs[0])
}
