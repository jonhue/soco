use crate::algorithms::graph_search::{Path, Paths};
use crate::config::{Config, IntegralConfig};
use crate::cost::{CostFn, SingleCostFn};
use crate::objective::scalar_movement;
use crate::problem::{
    IntegralSimplifiedSmoothedConvexOptimization,
    SimplifiedSmoothedConvexOptimization,
};
use crate::result::{Failure, Result};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::{assert, is_pow_of_2};
use noisy_float::prelude::*;
use num::ToPrimitive;
use std::collections::HashMap;

/// Vertice in the graph denoting time `t` and the value `j` at time `t`.
#[derive(Eq, Hash, PartialEq)]
struct Vertice(i32, i32);

#[derive(Clone)]
pub struct Options {
    /// Value at initial time `0`. Defaults to `0`.
    pub x_start: i32,
}
impl Default for Options {
    fn default() -> Self {
        Options { x_start: 0 }
    }
}

/// Graph-Based Optimal Algorithm
pub fn optimal_graph_search(
    p: IntegralSimplifiedSmoothedConvexOptimization<'_>,
    options: Options,
    inverted: bool,
) -> Result<Path> {
    assert(p.d == 1, Failure::UnsupportedProblemDimension(p.d))?;
    assert(is_pow_of_2(p.bounds[0]), Failure::MustBePowOf2)?;

    let k_init = if p.bounds[0] > 2 {
        (p.bounds[0] as f64).log(2.).floor() as u32 - 2
    } else {
        0
    };

    let mut result =
        find_schedule(&p, select_initial_rows(&p), inverted, options.x_start);

    if k_init > 0 {
        for k in k_init - 1..=0 {
            result = find_schedule(
                &p,
                select_next_rows(&p, &result.xs, k),
                inverted,
                options.x_start,
            );
        }
    }

    Ok(result)
}

/// Utility to transform a problem instance where `m` is not a power of `2` to an instance that is accepted by `optimal_graph_search`.
pub fn make_pow_of_2(
    p: IntegralSimplifiedSmoothedConvexOptimization,
) -> Result<IntegralSimplifiedSmoothedConvexOptimization> {
    assert(p.d == 1, Failure::UnsupportedProblemDimension(p.d))?;

    let m = 2_i32.pow((p.bounds[0] as f64).log(2.).ceil() as u32);

    Ok(SimplifiedSmoothedConvexOptimization {
        d: p.d,
        t_end: p.t_end,
        bounds: vec![m],
        switching_cost: p.switching_cost.clone(),
        hitting_cost: CostFn::stretch(
            1,
            p.t_end,
            SingleCostFn::certain(move |t, x: IntegralConfig| {
                if x[0] <= p.bounds[0] {
                    p.hit_cost(t, x)
                } else {
                    r64(x[0] as f64)
                        * (p.hit_cost(t, Config::new(p.bounds.clone()))
                            + f64::EPSILON)
                }
            }),
        ),
    })
}

fn select_initial_rows<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
) -> impl Fn(i32) -> Vec<i32> + 'a {
    move |_| (0..=4).map(|e| e * p.bounds[0] / 4).collect()
}

fn select_next_rows<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
    xs: &'a IntegralSchedule,
    k: u32,
) -> impl Fn(i32) -> Vec<i32> + 'a {
    move |t| {
        (-2..=2)
            .map(|e| xs[t as usize - 1][0] + e * 2_i32.pow(k))
            .filter(|&j| 0 <= j && j <= p.bounds[0])
            .collect()
    }
}

fn find_schedule(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    select_rows: impl Fn(i32) -> Vec<i32>,
    inverted: bool,
    x_start: i32,
) -> Path {
    let mut paths: Paths<Vertice> = HashMap::new();
    let initial_vertice = Vertice(0, x_start);
    let initial_path = Path {
        xs: Schedule::empty(),
        cost: 0.,
    };
    paths.insert(initial_vertice, initial_path);

    let mut prev_rows = vec![x_start];
    for t in 1..=p.t_end {
        let rows = select_rows(t);
        for &j in &rows {
            find_shortest_subpath(p, &mut paths, t, &prev_rows, j, inverted);
        }
        prev_rows = rows;
    }

    let mut result = Path {
        xs: Schedule::empty(),
        cost: f64::INFINITY,
    };
    for i in prev_rows {
        let path = &paths[&Vertice(p.t_end, i)];
        let cost = p.switching_cost[0] * scalar_movement(0, i, inverted) as f64;
        let picked_cost = path.cost + cost;
        if picked_cost < result.cost {
            result = Path {
                xs: path.xs.clone(),
                cost: picked_cost,
            };
        }
    }
    result
}

fn find_shortest_subpath(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    paths: &mut Paths<Vertice>,
    t: i32,
    from: &Vec<i32>,
    to: i32,
    inverted: bool,
) {
    let mut picked_source = 0;
    let mut picked_cost = f64::INFINITY;
    for &source in from {
        let prev_cost = paths[&Vertice(t - 1, source)].cost;
        let cost = build_cost(p, t, source, to, inverted);
        let new_cost = prev_cost + cost;
        if new_cost < picked_cost {
            picked_source = source;
            picked_cost = new_cost;
        };
    }
    update_paths(paths, t, picked_source, to, picked_cost);
}

fn build_cost(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    t: i32,
    i: i32,
    j: i32,
    inverted: bool,
) -> f64 {
    let hitting_cost = p.hit_cost(t, Config::single(j)).raw();
    let delta = ToPrimitive::to_f64(&scalar_movement(j, i, inverted)).unwrap();
    let switching_cost = p.switching_cost[0] * delta;
    hitting_cost + switching_cost
}

fn update_paths(paths: &mut Paths<Vertice>, t: i32, i: i32, j: i32, cost: f64) {
    let u = Vertice(t - 1, i);
    let v = Vertice(t, j);
    let prev_xs = &paths[&u].xs;
    let xs = prev_xs.extend(Config::single(j));

    paths.insert(v, Path { xs, cost });
}
