use crate::algorithms::capacity_provisioning::Bounded;
use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::Config;
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::{assert, project};

/// Backward-Recurrent Capacity Provisioning
pub fn brcp(
    p: FractionalSimplifiedSmoothedConvexOptimization<'_>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<f64>> {
    assert(!inverted, Failure::UnsupportedInvertedCost)?;
    assert(l.is_none(), Failure::UnsupportedLConstrainedMovement)?;
    assert(p.d == 1, Failure::UnsupportedProblemDimension(p.d))?;

    let mut xs = Schedule::empty();

    let mut x = 0.;
    for t in (1..=p.t_end).rev() {
        x = next(&p, alpha, t, x)?;
        xs.shift(Config::single(x));
    }

    Ok(PureOfflineResult { xs })
}

fn next(
    p: &FractionalSimplifiedSmoothedConvexOptimization<'_>,
    alpha: f64,
    t: i32,
    x: f64,
) -> Result<f64> {
    let l = p.find_lower_bound(alpha, t, 0, 0.)?;
    let u = p.find_upper_bound(alpha, t, 0, 0.)?;

    Ok(project(x, l, u))
}
