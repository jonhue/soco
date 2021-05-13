use crate::algorithms::capacity_provisioning::Bounded;
use crate::algorithms::online::uni_dimensional::lazy_capacity_provisioning::{
    find_initial_time, Memory,
};
use crate::online::{Online, OnlineSolution};
use crate::problem::DiscreteSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::{Config, DiscreteSchedule};
use crate::utils::{assert, project};

/// Discrete Lazy Capacity Provisioning
pub fn lcp(
    o: &Online<DiscreteSmoothedConvexOptimization<'_>>,
    xs: &DiscreteSchedule,
    ms: &Vec<Memory<i32>>,
) -> Result<OnlineSolution<Config<i32>, Memory<i32>>> {
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let t_start = find_initial_time(o.p.t_end, ms);

    let i = if xs.is_empty() {
        0
    } else {
        xs[xs.len() - 1][0]
    };
    let l = o.p.find_lower_bound(o.p.t_end, t_start)?;
    let u = o.p.find_upper_bound(o.p.t_end, t_start)?;
    let j = project(i, l, u);
    Ok(OnlineSolution(vec![j], Memory(l, u)))
}
