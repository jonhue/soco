use crate::config::Config;
use crate::convex_optimization::minimize;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use std::sync::Arc;

/// Memoryless Algorithm. Special case of Primal Online Balanced Descent.
pub fn memoryless(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    _: &(),
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let t = xs.t_end() + 1;
    let prev_x = if xs.is_empty() { 0. } else { xs.now()[0] };

    let x = next(o, t, prev_x)?;
    Ok(Step(Config::single(x), None))
}

/// Determines next `x` with a convex optimization.
fn next(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    prev_x: f64,
) -> Result<f64> {
    let bounds = vec![(0., o.p.bounds[0])];
    let objective = |xs: &[f64]| -> f64 {
        (o.p.hitting_cost)(t, Config::new(xs.to_vec())).unwrap()
    };
    let constraint = Arc::new(|xs: &[f64]| -> f64 {
        (xs[0] - prev_x).abs()
            - (o.p.hitting_cost)(t, Config::new(xs.to_vec())).unwrap() / 2.
    });

    let (xs, _) = minimize(objective, &bounds, None, vec![constraint], vec![])?;
    Ok(xs[0])
}
