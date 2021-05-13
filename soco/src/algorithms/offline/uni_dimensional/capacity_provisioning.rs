use itertools::Either::{Left, Right};

use crate::algorithms::capacity_provisioning::Bounded;
use crate::config::Config;
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::{FractionalSchedule, Schedule};
use crate::utils::{assert, project};

/// Backward-Recurrent Capacity Provisioning
pub fn bcp(
    p: &FractionalSmoothedConvexOptimization<'_>,
) -> Result<FractionalSchedule> {
    cp(p, false)
}

/// Forward-Recurrent Capacity Provisioning
pub fn fcp(
    p: &FractionalSmoothedConvexOptimization<'_>,
) -> Result<FractionalSchedule> {
    cp(p, true)
}

fn cp(
    p: &FractionalSmoothedConvexOptimization<'_>,
    forward: bool,
) -> Result<FractionalSchedule> {
    assert(p.d == 1, Error::UnsupportedProblemDimension)?;

    let mut xs = Schedule::empty();

    let mut x = 0.;
    let range = if forward {
        Left(1..=p.t_end)
    } else {
        Right((1..=p.t_end).rev())
    };
    for t in range {
        x = next(p, t, x)?;
        xs.shift(Config::single(x));
    }

    Ok(xs)
}

fn next(
    p: &FractionalSmoothedConvexOptimization<'_>,
    t: i32,
    x: f64,
) -> Result<f64> {
    let l = p.find_lower_bound(t, 0)?;
    let u = p.find_upper_bound(t, 0)?;

    Ok(project(x, l, u))
}
