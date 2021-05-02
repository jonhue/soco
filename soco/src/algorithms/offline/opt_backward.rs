use crate::algorithms::lcp::bounds::Bounded;
use crate::problem::ContinuousHomProblem;
use crate::result::Result;
use crate::schedule::ContinuousSchedule;
use crate::utils::project;

/// Optimal Backward Offline Algorithm
pub fn opt_backward(
    p: &ContinuousHomProblem<'_>,
) -> Result<ContinuousSchedule> {
    let mut xs = Vec::new();

    let mut x = 0.;
    for t in (1..=p.t_end).rev() {
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
