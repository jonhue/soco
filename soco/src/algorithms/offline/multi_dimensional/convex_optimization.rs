use crate::algorithms::offline::PureOfflineResult;
use crate::config::Config;
use crate::numerics::convex_optimization::find_minimizer;
use crate::objective::Objective;
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use noisy_float::prelude::*;

/// Convex Optimization
pub fn co(
    p: FractionalSmoothedConvexOptimization<'_>,
    _: (),
    inverted: bool,
) -> Result<PureOfflineResult<f64>> {
    assert(!inverted, Failure::UnsupportedInvertedCost)?;

    let (lower, upper): (Vec<_>, Vec<_>) = p.bounds.iter().cloned().unzip();
    let extended_lower = Schedule::build_raw(p.t_end, &Config::new(lower));
    let extended_upper = Schedule::build_raw(p.t_end, &Config::new(upper));
    let bounds = extended_lower
        .into_iter()
        .zip(extended_upper.into_iter())
        .collect();
    let objective = |raw_xs: &[f64]| {
        let xs = Schedule::from_raw(p.d, p.t_end, raw_xs);
        n64(p.objective_function(&xs).unwrap())
    };

    let (raw_xs, _) = find_minimizer(objective, &bounds)?;
    let xs = Schedule::from_raw(p.d, p.t_end, &raw_xs);
    Ok(PureOfflineResult { xs })
}
