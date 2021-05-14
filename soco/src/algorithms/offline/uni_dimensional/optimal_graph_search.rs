use std::collections::HashMap;
use std::sync::Arc;

use crate::algorithms::graph_search::{Path, Paths};
use crate::config::Config;
use crate::objective::movement;
use crate::problem::{
    IntegralSmoothedConvexOptimization, SmoothedConvexOptimization,
};
use crate::result::{Error, Result};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::{assert, is_pow_of_2};

/// Vertice in the graph denoting time `t` and the value `j` at time `t`.
#[derive(Eq, Hash, PartialEq)]
struct Vertice(i32, i32);

pub struct Options {
    /// Compute inverted cost.
    pub inverted: bool,
}

/// Graph-Based Optimal Integral Algorithm
pub fn optimal_graph_search(
    p: &'_ IntegralSmoothedConvexOptimization<'_>,
    options: &Options,
) -> Result<Path> {
    assert(p.d == 1, Error::UnsupportedProblemDimension)?;
    assert(is_pow_of_2(p.bounds[0]), Error::MustBePowOf2)?;

    let k_init = if p.bounds[0] > 2 {
        (p.bounds[0] as f64).log(2.).floor() as u32 - 2
    } else {
        0
    };

    let mut result =
        find_schedule(p, select_initial_rows(p), options.inverted)?;

    if k_init > 0 {
        for k in k_init - 1..=0 {
            result = find_schedule(
                p,
                select_next_rows(p, &result.0, k),
                options.inverted,
            )?;
        }
    }

    Ok(result)
}

/// Utility to transform a problem instance where `m` is not a power of `2` to an instance that is accepted by `iopt`.
pub fn make_pow_of_2<'a>(
    p: &'a IntegralSmoothedConvexOptimization<'a>,
) -> Result<IntegralSmoothedConvexOptimization<'a>> {
    assert(p.d == 1, Error::UnsupportedProblemDimension)?;

    let m = 2_i32.pow((p.bounds[0] as f64).log(2.).ceil() as u32);
    let hitting_cost = Arc::new(move |t, xs: Vec<i32>| {
        if xs[0] <= p.bounds[0] {
            (p.hitting_cost)(t, xs)
        } else {
            Some(
                xs[0] as f64
                    * ((p.hitting_cost)(t, (*p.bounds).to_vec()).unwrap()
                        + f64::EPSILON),
            )
        }
    });

    Ok(SmoothedConvexOptimization {
        d: p.d,
        t_end: p.t_end,
        bounds: vec![m],
        switching_cost: p.switching_cost.clone(),
        hitting_cost,
    })
}

fn select_initial_rows<'a>(
    p: &'a IntegralSmoothedConvexOptimization<'a>,
) -> impl Fn(i32) -> Vec<i32> + 'a {
    move |_| (0..=4).map(|e| e * p.bounds[0] / 4).collect()
}

fn select_next_rows<'a>(
    p: &'a IntegralSmoothedConvexOptimization<'a>,
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
    p: &IntegralSmoothedConvexOptimization<'_>,
    select_rows: impl Fn(i32) -> Vec<i32>,
    inverted: bool,
) -> Result<Path> {
    let mut paths: Paths<Vertice> = HashMap::new();
    let initial_vertice = Vertice(0, 0);
    let initial_path = Path(Schedule::empty(), 0.);
    paths.insert(initial_vertice, initial_path);

    let mut prev_rows = vec![0];
    for t in 1..=p.t_end {
        let rows = select_rows(t);
        for &j in &rows {
            find_shortest_subpath(p, &mut paths, t, &prev_rows, j, inverted)?;
        }
        prev_rows = rows;
    }

    let mut result = Path(Schedule::empty(), f64::INFINITY);
    for i in prev_rows {
        let path = paths
            .get(&Vertice(p.t_end, i))
            .ok_or(Error::PathsShouldBeCached)?;
        let cost = if inverted {
            p.switching_cost[0] * i as f64
        } else {
            0.
        };
        let picked_cost = path.1 + cost;
        if picked_cost < result.1 {
            result = Path(path.0.clone(), picked_cost);
        }
    }
    Ok(result)
}

fn find_shortest_subpath(
    p: &IntegralSmoothedConvexOptimization<'_>,
    paths: &mut Paths<Vertice>,
    t: i32,
    from: &Vec<i32>,
    to: i32,
    inverted: bool,
) -> Result<()> {
    let mut picked_source = 0;
    let mut picked_cost = f64::INFINITY;
    for &source in from {
        let prev_cost = paths
            .get(&Vertice(t - 1, source))
            .ok_or(Error::PathsShouldBeCached)?
            .1;
        let cost = build_cost(p, t, source, to, inverted)?;
        let new_cost = prev_cost + cost;
        if new_cost < picked_cost {
            picked_source = source;
            picked_cost = new_cost;
        };
    }
    update_paths(paths, t, picked_source, to, picked_cost)
}

fn build_cost(
    p: &IntegralSmoothedConvexOptimization<'_>,
    t: i32,
    i: i32,
    j: i32,
    inverted: bool,
) -> Result<f64> {
    let hitting_cost =
        (p.hitting_cost)(t, vec![j]).ok_or(Error::CostFnMustBeTotal)?;
    let switching_cost = p.switching_cost[0] * movement(j, i, inverted);
    Ok(hitting_cost + switching_cost)
}

fn update_paths(
    paths: &mut Paths<Vertice>,
    t: i32,
    i: i32,
    j: i32,
    c: f64,
) -> Result<()> {
    let u = Vertice(t - 1, i);
    let v = Vertice(t, j);
    let prev_xs = &paths.get(&u).ok_or(Error::PathsShouldBeCached)?.0;
    let xs = prev_xs.extend(Config::single(j));

    paths.insert(v, Path(xs, c));
    Ok(())
}
