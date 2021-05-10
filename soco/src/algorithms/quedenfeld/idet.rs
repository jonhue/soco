use crate::online::Online;
use crate::online::OnlineSolution;
use crate::problem::DiscreteSmoothedConstantOptimization;
use crate::result::{Error, Result};
use crate::schedule::DiscreteSchedule;
use crate::schedule::Step;
use crate::utils::{assert, total_bound};
use std::cmp::max;

/// Lane distribution at some time `t`.
pub type Memory = (Lanes, Horizons);

/// Maps each lane to the dimension it is "handled by" at some time `t`.
/// If value is `0`, there the lane is not "active".
pub type Lanes = Vec<i32>;

/// Maps each lane to a finite time horizon it stays "active" for unless replaced by another dimension.
pub type Horizons = Vec<i32>;

/// Deterministic Online Algorithm
pub fn idet<'a>(
    o: &'a Online<DiscreteSmoothedConstantOptimization>,
    xs: &DiscreteSchedule,
    ms: &Vec<Memory>,
) -> Result<OnlineSolution<Step<i32>, Memory>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;

    let t = xs.len() as i32 + 1;
    let bound = total_bound(&o.p.bounds) as usize;
    let optimal_lanes = find_optimal_lanes();
    let (prev_lanes, mut horizons) = if ms.is_empty() {
        (vec![0; bound], vec![0; bound])
    } else {
        ms[ms.len() - 1].clone()
    };

    let mut lanes = vec![0; bound];
    for k in 0..bound {
        if prev_lanes[k] < optimal_lanes[k] || t >= horizons[k] {
            lanes[k] = optimal_lanes[k];
            horizons[k] = t + next_time_horizon(
                &o.p.hitting_cost,
                &o.p.switching_cost,
                lanes[k],
            );
        } else {
            lanes[k] = prev_lanes[k];
            horizons[k] = max(
                horizons[k],
                t + next_time_horizon(
                    &o.p.hitting_cost,
                    &o.p.switching_cost,
                    lanes[k],
                ),
            );
        }
    }

    let step = collect_step(o.p.d, &lanes);
    Ok((step, (lanes, horizons)))
}

fn next_time_horizon(
    hitting_cost: &Vec<f64>,
    switching_cost: &Vec<f64>,
    k: i32,
) -> i32 {
    if k == 0 {
        0
    } else {
        (switching_cost[k as usize - 1] / hitting_cost[k as usize - 1]).floor()
            as i32
    }
}

fn collect_step(d: i32, lanes: &Lanes) -> Step<i32> {
    let mut step = vec![0; d as usize];
    for i in 0..lanes.len() {
        step[lanes[i] as usize] += 1;
    }
    step
}

// fn build_lanes(x: &Step<i32>, d: i32, bound: i32) -> Lanes {
//     let mut lanes = vec![0; bound as usize];
//     for k in 0..lanes.len() {
//         if k as i32 <= active_lanes(x, 1, d) {
//             for j in 1..=d {
//                 if active_lanes(x, j, d) >= k as i32 {
//                     lanes[k] = j;
//                 } else {
//                     continue;
//                 }
//             }
//         }
//     }
//     lanes
// }

// /// Sums step across dimension from `from` to `to`.
// fn active_lanes(x: &Step<i32>, from: i32, to: i32) -> i32 {
//     let mut result = 0;
//     for k in from..=to {
//         result += x[k as usize];
//     }
//     result
// }

fn find_optimal_lanes() -> Lanes {
    vec![]
}
