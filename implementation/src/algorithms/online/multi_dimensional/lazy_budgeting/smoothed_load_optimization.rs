use crate::algorithms::offline::multi_dimensional::optimal_graph_search::{
    optimal_graph_search, Options as OptimalGraphSearchOptions,
};
use crate::algorithms::offline::multi_dimensional::Vertice;
use crate::algorithms::offline::Cache;
use crate::algorithms::offline::OfflineAlgorithm;
use crate::algorithms::online::{IntegralStep, Online, Step};
use crate::config::{Config, IntegralConfig};
use crate::model::data_center::DataCenterModelOutputFailure;
use crate::problem::{
    DefaultGivenOnlineProblem, IntegralSmoothedLoadOptimization,
};
use crate::result::{Failure, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::{assert, sample_uniform};
use is_sorted::IsSorted;
use log::debug;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde_derive::{Deserialize, Serialize};
use std::cmp::max;

/// Lane distribution at some time `t`.
#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Memory {
    /// Lanes of the determined schedule.
    pub lanes: Lanes,
    /// Time horizons of each lane.
    pub horizons: Horizons,
    /// Factor for calculating next time horizons when using the randomized variant of the algorithm.
    pub gamma: f64,
    /// Cache of offline algorithm.
    cache: Option<Cache<Vertice>>,
}
impl
    DefaultGivenOnlineProblem<
        i32,
        IntegralSmoothedLoadOptimization,
        (),
        DataCenterModelOutputFailure,
    > for Memory
{
    fn default(o: &Online<IntegralSmoothedLoadOptimization>) -> Self {
        let bound: i32 = o.p.bounds.iter().sum();
        Memory {
            lanes: vec![0; bound as usize],
            horizons: vec![0; bound as usize],
            gamma: sample_gamma(),
            cache: None,
        }
    }
}

/// Utility to sample gamma for Randomized Lazy Budgeting.
///
/// Sample gamma once before running the algorithm.
fn sample_gamma() -> f64 {
    let r = sample_uniform(0., 1.);
    (r * (std::f64::consts::E - 1.) + 1.).ln()
}

/// Maps each lane to the dimension it is "handled by" at some time `t`.
/// If value is `0`, then the lane is not "active".
pub type Lanes = Vec<i32>;

/// Maps each lane to a finite time horizon it stays "active" for unless replaced by another dimension.
pub type Horizons = Vec<i32>;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Whether to use the randomized variant of the algorithm.
    pub randomized: bool,
}
impl Default for Options {
    fn default() -> Self {
        Options { randomized: false }
    }
}
#[pymethods]
impl Options {
    #[new]
    fn constructor(randomized: bool) -> Self {
        Options { randomized }
    }
}

/// Lazy Budgeting for Smoothed Load Optimization
pub fn lb(
    o: Online<IntegralSmoothedLoadOptimization>,
    t: i32,
    _: &IntegralSchedule,
    Memory {
        lanes: prev_lanes,
        horizons: prev_horizons,
        gamma,
        cache,
    }: Memory,
    Options { randomized }: Options,
) -> Result<IntegralStep<Memory>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let bound = o.p.bounds.iter().sum();
    debug!("starting with `m = {}`", bound);

    let (optimal_lanes, new_cache) =
        find_optimal_lanes(cache, o.p.clone(), bound)?;
    debug!("obtained optimal lanes: {:?}", optimal_lanes);

    let (lanes, horizons): (Vec<_>, Vec<_>) = (0..bound as usize)
        .into_par_iter()
        .map(|j| {
            if prev_lanes[j] < optimal_lanes[j] || t >= prev_horizons[j] {
                (
                    optimal_lanes[j],
                    t + next_time_horizon(
                        &o.p.hitting_cost,
                        &o.p.switching_cost,
                        optimal_lanes[j],
                        gamma,
                        randomized,
                    ),
                )
            } else {
                (
                    prev_lanes[j],
                    max(
                        prev_horizons[j],
                        t + next_time_horizon(
                            &o.p.hitting_cost,
                            &o.p.switching_cost,
                            optimal_lanes[j],
                            gamma,
                            randomized,
                        ),
                    ),
                )
            }
        })
        .unzip();
    debug!("updated lanes from {:?} to {:?}", prev_lanes, lanes);
    debug!(
        "updated horizons from {:?} to {:?}",
        prev_horizons, horizons
    );
    assert!(
        IsSorted::is_sorted(&mut horizons.iter().rev()),
        "horizons must be in descending order"
    );

    let config = collect_config(o.p.d, &lanes);
    Ok(Step(
        config,
        Some(Memory {
            lanes,
            horizons,
            gamma,
            cache: Some(new_cache),
        }),
    ))
}

fn next_time_horizon(
    hitting_cost: &Vec<f64>,
    switching_cost: &Vec<f64>,
    k: i32,
    gamma: f64,
    randomized: bool,
) -> i32 {
    if k == 0 {
        0
    } else {
        (if randomized { gamma } else { 1. } * switching_cost[k as usize - 1]
            / hitting_cost[k as usize - 1])
            .floor() as i32
    }
}

fn collect_config(d: i32, lanes: &Lanes) -> IntegralConfig {
    let mut config = Config::repeat(0, d);
    for i in 0..lanes.len() {
        if lanes[i] > 0 {
            config[lanes[i] as usize - 1] += 1;
        }
    }
    config
}

fn build_lanes(x: &IntegralConfig, d: i32, bound: i32) -> Lanes {
    let mut lanes = vec![0; bound as usize];
    for (k, lane) in lanes.iter_mut().enumerate() {
        #[allow(clippy::int_plus_one)]
        if k as i32 + 1 <= active_lanes(x, 1, d) {
            for j in 1..=d {
                #[allow(clippy::int_plus_one)]
                if active_lanes(x, j, d) >= k as i32 + 1 {
                    *lane = j;
                }
            }
        }
    }
    lanes
}

/// Sums step across dimension from `from` to `to`.
fn active_lanes(x: &IntegralConfig, from: i32, to: i32) -> i32 {
    (from..=to).map(|k| x[k as usize - 1]).sum()
}

fn find_optimal_lanes(
    cache: Option<Cache<Vertice>>,
    p: IntegralSmoothedLoadOptimization,
    bound: i32,
) -> Result<(Lanes, Cache<Vertice>)> {
    let d = p.d;
    let sblo_p = p.into_sblo();
    let ssco_p = sblo_p.into_ssco();
    let result = optimal_graph_search.solve(
        ssco_p,
        OptimalGraphSearchOptions { cache },
        Default::default(),
    )?;
    debug!(
        "obtained optimal schedule: {:?} (cost: `{}`)",
        result.path.xs, result.path.cost
    );
    let lanes = build_lanes(&result.path.xs.now(), d, bound);
    Ok((lanes, result.cache))
}
