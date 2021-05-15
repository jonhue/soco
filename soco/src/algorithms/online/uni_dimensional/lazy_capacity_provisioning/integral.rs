use crate::algorithms::capacity_provisioning::Bounded;
use crate::algorithms::offline::multi_dimensional::approx_graph_search::Options as ApproxOptions;
use crate::algorithms::online::uni_dimensional::lazy_capacity_provisioning::{
    find_initial_time, Memory,
};
use crate::config::Config;
use crate::online::{Online, Step};
use crate::problem::IntegralSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::{assert, project};

pub struct Options<'a> {
    /// Whether to to start optimization at reference time other than `0`.
    pub optimize_reference_time: bool,
    /// Whether to use an approximation to find the optimal schedule.
    pub use_approx: Option<&'a ApproxOptions>,
}

/// Integral Lazy Capacity Provisioning
pub fn lcp(
    o: &Online<IntegralSmoothedConvexOptimization<'_>>,
    xs: &mut IntegralSchedule,
    ms: &mut Vec<Memory<i32>>,
    options: &Options,
) -> Result<Step<i32, Memory<i32>>> {
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let t_start = if options.optimize_reference_time {
        find_initial_time(o.p.t_end, ms)
    } else {
        0
    };

    let i = if xs.is_empty() { 0 } else { xs.now()[0] };
    let l =
        o.p.find_lower_bound(o.p.t_end, t_start, options.use_approx)?;
    let u =
        o.p.find_upper_bound(o.p.t_end, t_start, options.use_approx)?;
    let j = project(i, l, u);
    Ok(Step(Config::single(j), Some(Memory(l, u))))
}
