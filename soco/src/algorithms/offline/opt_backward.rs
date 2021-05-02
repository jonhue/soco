use crate::algorithms::lcp::bounds::Bounded;
use crate::problem::ContinuousHomProblem;
use crate::result::{Error, Result};
use crate::schedule::ContinuousSchedule;
use crate::utils::{assert, project};

/// Optimal Backward Offline Algorithm
pub fn opt_backward(p: &ContinuousHomProblem<'_>) -> Result<(ContinuousSchedule, f64)> {
    let mut xs = Vec::new();

    let mut x = 0.;
    let mut cost = 0.;
    for t in (1..=p.t_end).rev() {
        let l = p.find_lower_bound(t, t)?;
        let u = p.find_upper_bound(t, t)?;
        if t == p.t_end {
            #[allow(clippy::float_cmp)]
            assert(l == u, Error::LcpBoundMismatch(l, u))?;
            cost = l;
        };

        x = project(x, l, u);
        xs.insert(0, x);
    }

    Ok((xs, cost))
}
