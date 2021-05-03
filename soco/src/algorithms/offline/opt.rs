use itertools::Either::{Left, Right};

use crate::algorithms::lcp::bounds::Bounded;
use crate::problem::ContinuousProblem;
use crate::result::{Error, Result};
use crate::schedule::ContinuousSchedule;
use crate::utils::{assert, project};

/// Optimal Backward Offline Algorithm
pub fn opt_backward(p: &ContinuousProblem<'_>) -> Result<ContinuousSchedule> {
    opt(p, false)
}

/// Optimal Forward Offline Algorithm
pub fn opt_forward(p: &ContinuousProblem<'_>) -> Result<ContinuousSchedule> {
    opt(p, true)
}

fn opt(p: &ContinuousProblem<'_>, forward: bool) -> Result<ContinuousSchedule> {
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

fn next(p: &ContinuousProblem<'_>, t: i32, x: f64) -> Result<f64> {
    let l = p.find_lower_bound(t, 0)?;
    let u = p.find_upper_bound(t, 0)?;

    Ok(project(x, l, u))
}
