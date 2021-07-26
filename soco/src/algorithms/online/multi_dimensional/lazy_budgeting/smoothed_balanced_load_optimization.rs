use crate::algorithms::offline::graph_search::Cache;
use crate::algorithms::offline::multi_dimensional::optimal_graph_search::{
    optimal_graph_search, Options as OptimalGraphSearchOptions,
};
use crate::algorithms::offline::multi_dimensional::Vertice;
use crate::algorithms::offline::{OfflineAlgorithm};
use crate::algorithms::online::{IntegralStep, Step};
use crate::config::{Config, IntegralConfig};
use crate::cost::{CostFn, SingleCostFn};
use crate::problem::{DefaultGivenProblem, IntegralSmoothedBalancedLoadOptimization, Online, SmoothedBalancedLoadOptimization};
use crate::result::{Failure, Result};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::assert;
use log::debug;
use noisy_float::prelude::*;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Derivative, Deserialize, Serialize)]
#[derivative(Debug)]
pub struct Memory<'a> {
    /// Maps each time `u` to the corresponding load
    load: Vec<i32>,
    /// Hitting costs for modified problem instance.
    #[serde(skip, default = "default_hitting_cost")]
    #[derivative(Debug = "ignore")]
    hitting_cost: Vec<CostFn<'a, f64>>,
    /// Schedule and memory of internally used algorithm.
    mod_m: (IntegralSchedule, Option<AlgBMemory>),
}
fn default_hitting_cost<'a>() -> Vec<CostFn<'a, f64>> {
    vec![]
}
impl<'a> DefaultGivenProblem<IntegralSmoothedBalancedLoadOptimization<'a>> for Memory<'a> {
    fn default(p: &IntegralSmoothedBalancedLoadOptimization) -> Self {
        Memory {
            load: vec![],
            hitting_cost: (0..p.d as usize).into_iter().map(|_| CostFn::empty()).collect(),
            mod_m: (Schedule::empty(), None),
        }
    }
}
impl IntoPy<PyObject> for Memory<'_> {
    fn into_py(self, py: Python) -> PyObject {
        self.mod_m.1.into_py(py)
    }
}

#[derive(Clone)]
pub struct Options {
    /// `epsilon > 0`. Defaults to `0.25`.
    pub epsilon: f64,
}
impl Default for Options {
    fn default() -> Self {
        Options { epsilon: 0.25 }
    }
}

/// Lazy Budgeting for Smoothed Balanced-Load Optimization
pub fn lb<'a>(
    o: Online<IntegralSmoothedBalancedLoadOptimization<'a>>,
    t: i32,
    _: &IntegralSchedule,
    Memory {
        mut load,
        mut hitting_cost,
        mod_m: (mut mod_xs, mod_prev_m),
    }: Memory<'a>,
    options: Options,
) -> Result<IntegralStep<Memory<'a>>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    // determine number of sub time slots
    let n = determine_sub_time_slots(&o.p, t, options.epsilon)?;
    debug!("using {} sub time slots", n);

    // construct modified problem instance and update loads
    let u_init = load.len() as i32 + 1;
    let u_end = u_init + n;
    debug!("constructing modified problem instance from sub time slot {} to sub time slot {}", u_init, u_end);
    let mut mod_o = Online {
        w: 0,
        p: modify_problem(o.p, &mut load, &mut hitting_cost, t, n, u_init, u_end)?,
    };
    debug!("constructed modified problem instance: {:?}", mod_o);

    // execute `n` time slots of algorithm B on modified problem instance
    let mod_prev_m = mod_o.offline_stream_from(
        &alg_b,
        u_end,
        (),
        &mut mod_xs,
        mod_prev_m,
    )?;
    debug!("streamed modified problem and obtained schedule: {:?}", mod_xs);

    debug!("determining optimal config between sub time slots {} and {}", u_init, u_end);
    let config = determine_config(&mod_o.p, &mod_xs, u_init, u_end);
    debug!("found optimal config {:?}", config);

    Ok(Step(
        config,
        Some(Memory {
            load,
            hitting_cost,
            mod_m: (mod_xs, mod_prev_m),
        }),
    ))
}

/// Calculates the number of sub time slots for some time slot `t`.
fn determine_sub_time_slots(
    p: &IntegralSmoothedBalancedLoadOptimization,
    t: i32,
    epsilon: f64,
) -> Result<i32> {
    let max_fract = (0..p.d as usize)
        .map(|k| -> N64 {
            let l = p.hitting_cost[k].call_certain(t, 0.);
            l / n64(p.switching_cost[k])
        })
        .max()
        .unwrap()
        .raw();
    Ok((p.d as f64 / epsilon * max_fract).ceil() as i32)
}

/// Builds the modified problem instance and extends the cached sequence of loads.
fn modify_problem<'a>(
    p: IntegralSmoothedBalancedLoadOptimization<'a>,
    load: &mut Vec<i32>,
    hitting_cost: &mut Vec<CostFn<'a, f64>>,
    t: i32,
    n: i32,
    u_init: i32,
    u_end: i32,
) -> Result<IntegralSmoothedBalancedLoadOptimization<'a>> {
    assert!(u_end - u_init == n, "number of sub time slots inconsistent");

    (0..p.d as usize)
        .for_each(|k| {
            let raw_hitting_cost = p.hitting_cost[k].clone();
            hitting_cost[k].add(u_init, SingleCostFn::certain(move |u, x| {
                assert!(
                    u_init <= u && u <= u_end,
                    "sub time slot is outside valid sub time slot window"
                );
                raw_hitting_cost.call_certain(t, x) / n as f64
            }));
        });

    load.extend(vec![p.load[t as usize - 1]; n as usize + 1]);
    assert!(
        load.len() as i32 == u_end,
        "loads inconsistent with number of sub time slots"
    );

    Ok(SmoothedBalancedLoadOptimization {
        d: p.d,
        t_end: u_init,
        bounds: p.bounds,
        switching_cost: p.switching_cost,
        hitting_cost: hitting_cost.clone(),
        load: load.clone(),
    })
}

fn determine_config(
    mod_p: &IntegralSmoothedBalancedLoadOptimization,
    mod_xs: &IntegralSchedule,
    u_init: i32,
    u_end: i32,
) -> IntegralConfig {
    let (min_u, _) = (u_init + 1..=u_end).into_iter().fold(
        (
            u_init,
            mod_p
                .clone()
                .hit_cost(u_init, mod_xs.get(u_init).unwrap().clone()),
        ),
        |(min_u, min_c), u| {
            let c = mod_p.clone().hit_cost(u, mod_xs.get(u).unwrap().clone());
            if c < min_c {
                (u, c)
            } else {
                (min_u, min_c)
            }
        },
    );
    mod_xs.get(min_u).unwrap().clone()
}

#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct AlgBMemory {
    /// Maps dimension to the number of added instances for some sub time slot `u`.
    init_times: Vec<Vec<i32>>,
    /// Cache of offline algorithm.
    cache: Option<Cache<Vertice>>,
}
impl Default for AlgBMemory {
    fn default() -> Self {
        AlgBMemory {
            init_times: vec![],
            cache: None,
        }
    }
}

fn alg_b(
    o: Online<IntegralSmoothedBalancedLoadOptimization>,
    t: i32,
    xs: &IntegralSchedule,
    AlgBMemory {
        mut init_times,
        cache,
    }: AlgBMemory,
    _: (),
) -> Result<IntegralStep<AlgBMemory>> {
    let (opt_x, new_cache) = find_optimal_config(cache, o.p.clone())?;
    let prev_x = if xs.is_empty() {
        Config::repeat(0, o.p.d)
    } else {
        xs.now()
    };

    let (new_init_times, x) = (0..o.p.d as usize)
        .into_iter()
        .map(|k| {
            let j = prev_x[k]
                - deactivated_quantity(
                    &o.p.hitting_cost[k],
                    o.p.switching_cost[k],
                    &init_times,
                    t,
                    k,
                );
            if j < opt_x[k] {
                (opt_x[k] - j, opt_x[k])
            } else {
                (0, j)
            }
        })
        .unzip();

    init_times.push(new_init_times);
    let m = AlgBMemory {
        init_times,
        cache: Some(new_cache),
    };

    Ok(Step(Config::new(x), Some(m)))
}

fn deactivated_quantity(
    hitting_cost: &CostFn<'_, f64>,
    switching_cost: f64,
    init_times: &Vec<Vec<i32>>,
    t_now: i32,
    k: usize,
) -> i32 {
    (1..=t_now - 1)
        .into_par_iter()
        .map(|t| {
            let cum_l =
                cumulative_idle_hitting_cost(hitting_cost, t + 1, t_now - 1);
            let l = hitting_cost.call_certain(t_now, 0.).raw();

            if cum_l <= switching_cost && switching_cost < cum_l + l {
                init_times[t as usize - 1][k]
            } else {
                0
            }
        })
        .sum()
}

fn cumulative_idle_hitting_cost(
    hitting_cost: &CostFn<'_, f64>,
    from: i32,
    to: i32,
) -> f64 {
    (from..=to)
        .into_iter()
        .map(|t| hitting_cost.call_certain(t, 0.).raw())
        .sum()
}

fn find_optimal_config(
    cache: Option<Cache<Vertice>>,
    p: IntegralSmoothedBalancedLoadOptimization,
) -> Result<(IntegralConfig, Cache<Vertice>)> {
    let ssco_p = p.into_ssco();
    let result = optimal_graph_search.solve(
        ssco_p,
        OptimalGraphSearchOptions { cache },
        Default::default(),
    )?;
    Ok((result.path.xs.now(), result.cache))
}
