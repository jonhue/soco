use std::collections::HashMap;
use std::sync::Arc;

use crate::problem::{
    DiscreteSmoothedConvexOptimization, SmoothedConvexOptimization,
};
use crate::result::{Error, Result};
use crate::schedule::DiscreteSchedule;
use crate::utils::{assert, is_pow_of_2, pos};

/// The minimal cost from some initial vertice alongside the shortest path to the final vertice.
type Path = (DiscreteSchedule, f64);
/// Maps a vertice to its minimal cost from some initial vertice alongside the shortest path.
type Paths = HashMap<(i32, i32), Path>;

/// Optimal Discrete Deterministic Offline Algorithm
pub fn iopt(p: &'_ DiscreteSmoothedConvexOptimization<'_>) -> Result<Path> {
    _iopt(p, false)
}

/// Inverted Optimal Discrete Deterministic Offline Algorithm
pub fn inverted_iopt(
    p: &'_ DiscreteSmoothedConvexOptimization<'_>,
) -> Result<Path> {
    _iopt(p, true)
}

fn _iopt<'a>(
    p: &'a DiscreteSmoothedConvexOptimization<'a>,
    inverted: bool,
) -> Result<Path> {
    assert(p.d == 1, Error::UnsupportedProblemDimension)?;
    assert(is_pow_of_2(p.bounds[0]), Error::MustBePowOf2)?;

    let k_init = if p.bounds[0] > 2 {
        (p.bounds[0] as f64).log(2.).floor() as u32 - 2
    } else {
        0
    };

    let mut result = find_schedule(p, select_initial_rows(p), inverted)?;

    if k_init > 0 {
        for k in k_init - 1..=0 {
            result =
                find_schedule(p, select_next_rows(p, &result.0, k), inverted)?;
        }
    }

    Ok(result)
}

/// Utility to transform a problem instance where `m` is not a power of `2` to an instance that is accepted by `iopt`.
pub fn make_pow_of_2<'a>(
    p: &'a DiscreteSmoothedConvexOptimization<'a>,
) -> Result<DiscreteSmoothedConvexOptimization<'a>> {
    assert(p.d == 1, Error::UnsupportedProblemDimension)?;

    let m = 2_i32.pow((p.bounds[0] as f64).log(2.).ceil() as u32);
    let hitting_cost = Arc::new(move |t, xs: &Vec<i32>| {
        if xs[0] <= p.bounds[0] {
            (p.hitting_cost)(t, xs)
        } else {
            Some(
                xs[0] as f64
                    * ((p.hitting_cost)(t, &p.bounds).unwrap()
                        + std::f64::EPSILON),
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
    p: &'a DiscreteSmoothedConvexOptimization<'a>,
) -> impl Fn(i32) -> Vec<i32> + 'a {
    move |_| (0..=4).map(|e| e * p.bounds[0] / 4).collect()
}

fn select_next_rows<'a>(
    p: &'a DiscreteSmoothedConvexOptimization<'a>,
    xs: &'a DiscreteSchedule,
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
    p: &DiscreteSmoothedConvexOptimization<'_>,
    select_rows: impl Fn(i32) -> Vec<i32>,
    inverted: bool,
) -> Result<Path> {
    let mut paths: Paths = HashMap::new();
    paths.insert((0, 0), (vec![], 0.));

    let mut prev_rows = vec![0];
    for t in 1..=p.t_end {
        let rows = select_rows(t);
        for &j in &rows {
            find_shortest_path(p, &mut paths, t, &prev_rows, j, inverted)?;
        }
        prev_rows = rows;
    }

    let mut result = &(vec![], f64::INFINITY);
    for i in prev_rows {
        let path =
            paths.get(&(p.t_end, i)).ok_or(Error::PathsShouldBeCached)?;
        if path.1 < result.1 {
            result = path;
        }
    }

    Ok(result.clone())
}

fn find_shortest_path(
    p: &DiscreteSmoothedConvexOptimization<'_>,
    paths: &mut Paths,
    t: i32,
    from: &Vec<i32>,
    to: i32,
    inverted: bool,
) -> Result<()> {
    let mut picked_source = 0;
    let mut picked_cost = f64::INFINITY;
    for &source in from {
        let prev_cost = paths
            .get(&(t - 1, source))
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
    p: &DiscreteSmoothedConvexOptimization<'_>,
    t: i32,
    i: i32,
    j: i32,
    inverted: bool,
) -> Result<f64> {
    let hitting_cost =
        (p.hitting_cost)(t, &vec![j]).ok_or(Error::CostFnMustBeTotal)?;
    let switching_cost =
        p.switching_cost[0] * pos(if inverted { i - j } else { j - i }) as f64;
    Ok(hitting_cost + switching_cost)
}

fn update_paths(
    paths: &mut Paths,
    t: i32,
    i: i32,
    j: i32,
    c: f64,
) -> Result<()> {
    let u = (t - 1, i);
    let v = (t, j);
    let prev_xs = &paths.get(&u).ok_or(Error::PathsShouldBeCached)?.0;
    let xs = [&prev_xs[..], &[vec![j]]].concat();

    paths.insert(v, (xs, c));
    Ok(())
}
