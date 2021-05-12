use itertools::Either::{Left, Right};

use crate::algorithms::capacity_provisioning::Bounded;
use crate::problem::ContinuousSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::ContinuousSchedule;
use crate::utils::{assert, project};

/// Backward-Recurrent Capacity Provisioning
pub fn bcp(
    p: &ContinuousSmoothedConvexOptimization<'_>,
) -> Result<ContinuousSchedule> {
    cp(p, false)
}

/// Forward-Recurrent Capacity Provisioning
pub fn fcp(
    p: &ContinuousSmoothedConvexOptimization<'_>,
) -> Result<ContinuousSchedule> {
    cp(p, true)
}

fn cp(
    p: &ContinuousSmoothedConvexOptimization<'_>,
    forward: bool,
) -> Result<ContinuousSchedule> {
    assert(p.d == 1, Error::UnsupportedProblemDimension)?;

    let mut xs = Vec::new();

    let mut x = 0.;
    let range = if forward {
        Left(1..=p.t_end)
    } else {
        Right((1..=p.t_end).rev())
    };
    for t in range {
        x = next(p, t, x)?;
        xs.insert(0, vec![x]);
    }

    Ok(xs)
}

fn next(
    p: &ContinuousSmoothedConvexOptimization<'_>,
    t: i32,
    x: f64,
) -> Result<f64> {
    let l = p.find_lower_bound(t, 0)?;
    let u = p.find_upper_bound(t, 0)?;

    Ok(project(x, l, u))
}
