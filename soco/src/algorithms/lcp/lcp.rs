use crate::algorithms::lcp::bounds::Bounded;
use crate::online::{Online, OnlineSolution};
use crate::problem::ContinuousHomProblem;
use crate::result::Result;
use crate::schedule::ContinuousSchedule;
use crate::utils::project;

/// Lower and upper bound at some time t.
pub type Memory<T> = (T, T);

/// (Continuous) Lazy Capacity Provisioning
pub fn lcp(
    o: &Online<ContinuousHomProblem<'_>>,
    xs: &ContinuousSchedule,
    _: &Vec<Memory<f64>>,
) -> Result<OnlineSolution<f64, Memory<f64>>> {
    let i = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };
    let l = o.p.find_lower_bound(o.p.t_end, o.p.t_end + o.w)?;
    let u = o.p.find_upper_bound(o.p.t_end, o.p.t_end + o.w)?;
    let j = project(i, l, u);
    Ok((j, (l, u)))
}
