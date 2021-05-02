use crate::algorithms::lcp::bounds::Bounded;
use crate::algorithms::lcp::{find_initial_time, Memory};
use crate::online::{Online, OnlineSolution};
use crate::problem::DiscreteHomProblem;
use crate::result::Result;
use crate::schedule::DiscreteSchedule;
use crate::utils::project;

/// Integer Lazy Capacity Provisioning
pub fn ilcp(
    o: &Online<DiscreteHomProblem<'_>>,
    xs: &DiscreteSchedule,
    ms: &Vec<Memory<i32>>,
) -> Result<OnlineSolution<i32, Memory<i32>>> {
    let t_start = find_initial_time(o.p.t_end, ms);

    let i = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
    let l = o.p.find_lower_bound(o.p.t_end, t_start)?;
    let u = o.p.find_upper_bound(o.p.t_end, t_start)?;
    let j = project(i, l, u);
    Ok((j, (l, u)))
}
