use crate::algorithms::capacity_provisioning::Bounded;
use crate::config::Config;
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::{FractionalSchedule, Schedule};
use crate::utils::{assert, project};

/// Backward-Recurrent Capacity Provisioning
pub fn brcp(
    p: FractionalSimplifiedSmoothedConvexOptimization<'_>,
    _: (),
    inverted: bool,
) -> Result<FractionalSchedule> {
    assert(!inverted, Failure::UnsupportedInvertedCost)?;
    assert(p.d == 1, Failure::UnsupportedProblemDimension(p.d))?;

    let mut xs = Schedule::empty();

    let mut x = 0.;
    for t in (1..=p.t_end).rev() {
        x = next(&p, t, x)?;
        xs.shift(Config::single(x));
    }

    Ok(xs)
}

fn next(
    p: &FractionalSimplifiedSmoothedConvexOptimization<'_>,
    t: i32,
    x: f64,
) -> Result<f64> {
    let l = p.find_lower_bound(t, 0, 0.)?;
    let u = p.find_upper_bound(t, 0, 0.)?;

    Ok(project(x, l, u))
}
