use crate::algorithms::capacity_provisioning::Bounded;
use crate::algorithms::online::uni_dimensional::lazy_capacity_provisioning::{
    find_initial_time, Memory,
};
use crate::config::Config;
use crate::online::{Online, OnlineSolution};
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::{assert, project};

/// Fractional Lazy Capacity Provisioning
pub fn lcp(
    o: &Online<FractionalSmoothedConvexOptimization<'_>>,
    xs: &FractionalSchedule,
    ms: &Vec<Memory<f64>>,
    _: &(),
) -> Result<OnlineSolution<f64, Memory<f64>>> {
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let t_start = find_initial_time(o.p.t_end, ms);

    let i = if xs.is_empty() { 0. } else { xs.now()[0] };
    let l = o.p.find_lower_bound(o.p.t_end, t_start, None)?;
    let u = o.p.find_upper_bound(o.p.t_end, t_start, None)?;
    let j = project(i, l, u);
    Ok(OnlineSolution(Config::single(j), Memory(l, u)))
}
