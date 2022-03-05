use crate::algorithms::capacity_provisioning::Bounded;
use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::{assert, project};

/// Backward-Recurrent Capacity Provisioning
pub fn brcp<C, D>(
    p: FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<f64>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
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

fn next<C, D>(
    p: &FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>,
    alpha: f64,
    t: i32,
    x: f64,
) -> Result<f64>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let lower = p.find_alpha_unfair_lower_bound(alpha, 0, t, 0, 0.)?;
    let upper = p.find_alpha_unfair_upper_bound(alpha, 0, t, 0, 0.)?;
    println!("[{}; {}]", lower, upper);

    Ok(project(x, lower, upper))
}
