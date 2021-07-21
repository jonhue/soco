use crate::algorithms::offline::multi_dimensional::optimal_graph_search::optimal_graph_search;
use crate::algorithms::offline::{
    OfflineAlgorithm, OfflineOptions, OfflineResult,
};
use crate::algorithms::online::{IntegralStep, Step};
use crate::config::{Config, IntegralConfig};
use crate::cost::{CostFn, SingleCostFn};
use crate::problem::{
    IntegralSmoothedBalancedLoadOptimization, Online,
    SmoothedBalancedLoadOptimization,
};
use crate::result::{Failure, Result};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::assert;
use noisy_float::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct Memory {
    /// Maps each time `u` to the corresponding load; length
    load: Vec<i32>,
    /// Schedule and memory of internally used algorithm.
    mod_m: (IntegralSchedule, Option<AlgBMemory>),
}
impl Default for Memory {
    fn default() -> Self {
        Memory {
            load: vec![],
            mod_m: (Schedule::empty(), None),
        }
    }
}

/// Maps dimension to the number of added instances for some sub time slot `u`.
type AlgBMemory = Vec<Vec<i32>>;

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
pub fn lb(
    o: Online<IntegralSmoothedBalancedLoadOptimization>,
    t: i32,
    xs: &IntegralSchedule,
    Memory {
        mut load,
        mod_m: (mut mod_xs, mod_prev_m),
    }: Memory,
    options: Options,
) -> Result<IntegralStep<Memory>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    // determine number of sub time slots
    let n = determine_sub_time_slots(&o.p, t, options.epsilon)?;

    // construct modified problem instance and update loads
    let u_init = load.len() as i32 + 1;
    let u_end = u_init + n;
    let mut mod_o = Online {
        w: 0,
        p: modify_problem(o.p, &mut load, t, n, u_init, u_end)?,
    };

    // execute `n` time slots of algorithm B on modified problem instance
    let mod_prev_m = mod_o.offline_stream_from(
        &alg_b,
        u_end,
        (),
        &mut mod_xs,
        mod_prev_m,
    )?;

    // collect the resulting configuration
    let config = determine_config(&mod_o.p, xs, u_init, u_end);
    Ok(Step(
        config,
        Some(Memory {
            load,
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
            let l = p.hitting_cost[k].call(t, 0., &p.bounds[k]);
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
    t: i32,
    n: i32,
    u_init: i32,
    u_end: i32,
) -> Result<IntegralSmoothedBalancedLoadOptimization<'a>> {
    assert!(u_end - u_init == n, "number of sub time slots inconsistent");

    let hitting_cost = (0..p.d as usize)
        .map(|k| {
            let hitting_cost = p.hitting_cost[k].clone();
            let bounds = p.bounds[k];
            CostFn::new(
                u_init,
                SingleCostFn::certain(move |u, x| {
                    assert!(
                        u_init <= u && u <= u_end,
                        "sub time slot is outside valid sub time slot window"
                    );
                    hitting_cost.call(t, x, &bounds) / n as f64
                }),
            )
        })
        .collect();

    load.extend(vec![p.load[t as usize - 1]; n as usize]);
    assert!(
        load.len() as i32 == u_end,
        "loads inconsistent with number of sub time slots"
    );

    Ok(SmoothedBalancedLoadOptimization {
        d: p.d,
        t_end: u_init,
        bounds: p.bounds,
        switching_cost: p.switching_cost,
        hitting_cost,
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

fn alg_b(
    o: Online<IntegralSmoothedBalancedLoadOptimization>,
    t: i32,
    xs: &IntegralSchedule,
    mut ms: AlgBMemory,
    _: (),
) -> Result<IntegralStep<AlgBMemory>> {
    let opt_x = find_optimal_config(o.p.clone())?;
    let prev_x = if xs.is_empty() {
        Config::repeat(0, o.p.d)
    } else {
        xs.now()
    };

    let (m, x) = (0..o.p.d as usize)
        .into_iter()
        .map(|k| {
            let j = prev_x[k]
                - deactivated_quantity(
                    o.p.bounds[k],
                    &o.p.hitting_cost[k],
                    o.p.switching_cost[k],
                    &ms,
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

    ms.push(m);
    Ok(Step(Config::new(x), Some(ms)))
}

fn deactivated_quantity(
    bound: i32,
    hitting_cost: &CostFn<'_, f64>,
    switching_cost: f64,
    ms: &AlgBMemory,
    t_now: i32,
    k: usize,
) -> i32 {
    (1..=t_now - 1)
        .into_par_iter()
        .map(|t| {
            let cum_l = cumulative_idle_hitting_cost(
                bound,
                hitting_cost,
                t + 1,
                t_now - 1,
            );
            let l = hitting_cost.call(t_now, 0., &bound).raw();

            if cum_l <= switching_cost && switching_cost < cum_l + l {
                ms[t as usize - 1][k]
            } else {
                0
            }
        })
        .sum()
}

fn cumulative_idle_hitting_cost(
    bound: i32,
    hitting_cost: &CostFn<'_, f64>,
    from: i32,
    to: i32,
) -> f64 {
    (from..=to)
        .into_iter()
        .map(|t| hitting_cost.call(t, 0., &bound).raw())
        .sum()
}

fn find_optimal_config(
    p: IntegralSmoothedBalancedLoadOptimization,
) -> Result<IntegralConfig> {
    let ssco_p = p.into_ssco();
    let result =
        optimal_graph_search.solve(ssco_p, (), OfflineOptions::default())?;
    Ok(result.xs().now())
}
