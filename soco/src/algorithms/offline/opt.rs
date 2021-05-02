use itertools::Either::{Left, Right};

use crate::algorithms::lcp::bounds::Bounded;
use crate::problem::ContinuousHomProblem;
use crate::result::Result;
use crate::schedule::ContinuousSchedule;
use crate::utils::project;

/// Optimal Backward Offline Algorithm
pub fn opt_backward(
    p: &ContinuousHomProblem<'_>,
) -> Result<ContinuousSchedule> {
    opt(p, false)
}

/// Optimal Forward Offline Algorithm
pub fn opt_forward(p: &ContinuousHomProblem<'_>) -> Result<ContinuousSchedule> {
    opt(p, true)
}

fn opt(
    p: &ContinuousHomProblem<'_>,
    forward: bool,
) -> Result<ContinuousSchedule> {
    let mut xs = Vec::new();

    let mut x = 0.;
    let range = if forward {
        Left(1..=p.t_end)
    } else {
        Right((1..=p.t_end).rev())
    };
    for t in range {
        x = next(p, t, x)?;
        xs.insert(0, x);
    }

    Ok(xs)
}

fn next(p: &ContinuousHomProblem<'_>, t: i32, x: f64) -> Result<f64> {
    let l = p.find_lower_bound(t, t)?;
    let u = p.find_upper_bound(t, t)?;

    Ok(project(x, l, u))
}
