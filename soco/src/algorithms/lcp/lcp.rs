use crate::algorithms::lcp::bounds::Bounded;
use crate::algorithms::lcp::{find_initial_time, Memory};
use crate::online::{Online, OnlineSolution};
use crate::problem::ContinuousHomProblem;
use crate::result::Result;
use crate::schedule::ContinuousSchedule;
use crate::utils::project;

/// (Continuous) Lazy Capacity Provisioning
pub fn lcp(
    o: &Online<ContinuousHomProblem<'_>>,
    xs: &ContinuousSchedule,
    ms: &Vec<Memory<f64>>,
) -> Result<OnlineSolution<f64, Memory<f64>>> {
    let t_start = find_initial_time(o.p.t_end, ms);

    let i = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };
    let l = o.p.find_lower_bound(o.p.t_end, t_start)?;
    let u = o.p.find_upper_bound(o.p.t_end, t_start)?;
    let j = project(i, l, u);
    Ok((j, (l, u)))
}
