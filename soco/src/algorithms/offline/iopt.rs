use ordered_float::OrderedFloat;
use pathfinding::directed::dijkstra::dijkstra;
use std::collections::HashMap;
use std::sync::Arc;

use crate::problem::{DiscreteHomProblem, HomProblem};
use crate::result::{Error, Result};
use crate::schedule::DiscreteSchedule;
use crate::utils::{assert, is_pow_of_2, pos};

/// Represents a vertice `v_{t, j}` where the `t ~ time` and `j ~ #servers`.
type Vertice = (i32, i32);
/// Represents the length (cost) of an edge.
type Cost = OrderedFloat<f64>;
/// Maps a vertice to all its neighbors with some cost.
type Neighbors = HashMap<Vertice, Vec<(Vertice, Cost)>>;

/// Discrete Deterministic Offline Algorithm
pub fn iopt(p: &'_ DiscreteHomProblem<'_>) -> Result<(DiscreteSchedule, f64)> {
    _iopt(p, false)
}

/// Inverted Discrete Deterministic Offline Algorithm
pub fn inverted_iopt(
    p: &'_ DiscreteHomProblem<'_>,
) -> Result<(DiscreteSchedule, f64)> {
    _iopt(p, true)
}

fn _iopt<'a>(
    p: &'a DiscreteHomProblem<'a>,
    inverted: bool,
) -> Result<(DiscreteSchedule, f64)> {
    assert(is_pow_of_2(p.m), Error::MustBePowOf2)?;

    let neighbors = build_neighbors(p, inverted)?;

    let k_init = if p.m > 2 {
        (p.m as f64).log(2.).floor() as u32 - 2
    } else {
        0
    };

    let initial_neighbors = select_initial_neighbors(p, &neighbors);
    let mut result = find_schedule(p, initial_neighbors);

    if k_init > 0 {
        for k in k_init - 1..=0 {
            let next_neighbors =
                select_next_neighbors(p, &result.0, &neighbors, k);
            result = find_schedule(p, next_neighbors);
        }
    }

    Ok(result)
}

/// Utility to transform a problem instance where `m` is not a power of `2` to an instance that is accepted by `iopt`.
pub fn make_pow_of_2<'a>(
    p: &'a DiscreteHomProblem<'a>,
) -> DiscreteHomProblem<'a> {
    let m = 2_i32.pow((p.m as f64).log(2.).ceil() as u32);
    let f = Arc::new(move |t, x| {
        if x <= p.m {
            (p.f)(t, x)
        } else {
            Some(x as f64 * ((p.f)(t, p.m).unwrap() + std::f64::EPSILON))
        }
    });

    HomProblem {
        m,
        t_end: p.t_end,
        beta: p.beta,
        f,
    }
}

fn build_neighbors(
    p: &DiscreteHomProblem<'_>,
    inverted: bool,
) -> Result<Neighbors> {
    let mut neighbors = HashMap::new();
    neighbors.insert((0, 0), build_edges(p, 0, 0, inverted)?);
    for t in 1..=p.t_end {
        for i in 0..p.m {
            neighbors.insert((t, i), build_edges(p, t, i, inverted)?);
        }
    }
    Ok(neighbors)
}

fn build_edges(
    p: &DiscreteHomProblem<'_>,
    t: i32,
    i: i32,
    inverted: bool,
) -> Result<Vec<(Vertice, Cost)>> {
    if t == p.t_end {
        Ok(vec![((p.t_end + 1, 0), OrderedFloat(0.))])
    } else {
        vec![0; p.m as usize]
            .iter()
            .enumerate()
            .map(|(j, _)| {
                Ok((
                    (t + 1, j as i32),
                    build_cost(p, t + 1, i, j as i32, inverted)?,
                ))
            })
            .collect()
    }
}

fn build_cost(
    p: &DiscreteHomProblem<'_>,
    t: i32,
    i: i32,
    j: i32,
    inverted: bool,
) -> Result<Cost> {
    Ok(OrderedFloat(
        p.beta * pos(if inverted { i - j } else { j - i }) as f64
            + (p.f)(t, j).ok_or(Error::CostFnMustBeTotal)?,
    ))
}

fn find_schedule(
    p: &DiscreteHomProblem<'_>,
    neighbors: impl Fn(&Vertice) -> Vec<(Vertice, Cost)>,
) -> (DiscreteSchedule, f64) {
    let result = dijkstra(&(0, 0), neighbors, |&(t, j): &Vertice| {
        (t, j) == (p.t_end + 1, 0)
    });
    let (mut xs, cost) = result.expect("there should always be a path");
    xs.remove(0);
    xs.remove(xs.len() - 1);
    (xs.into_iter().map(|(_, j)| j).collect(), cost.into_inner())
}

fn select_initial_neighbors<'a>(
    p: &DiscreteHomProblem<'a>,
    neighbors: &'a Neighbors,
) -> impl Fn(&Vertice) -> Vec<(Vertice, Cost)> + 'a {
    let acceptable_successors: Vec<i32> =
        (0..=4).map(|e| e * p.m / 4).collect();
    select_neighbors(neighbors, move |&(_, j)| {
        acceptable_successors.contains(&j)
    })
}

fn select_next_neighbors<'a>(
    p: &'a DiscreteHomProblem<'a>,
    xs: &DiscreteSchedule,
    neighbors: &'a Neighbors,
    k: u32,
) -> impl Fn(&Vertice) -> Vec<(Vertice, Cost)> + 'a {
    let acceptable_successors: Vec<Vec<i32>> = (1..=p.t_end)
        .map(|t| {
            (-2..=2)
                .map(|e| xs[t as usize - 1] + e * 2_i32.pow(k))
                .collect()
        })
        .collect();
    select_neighbors(neighbors, move |&(t, j)| {
        t == p.t_end + 1 || acceptable_successors[t as usize - 1].contains(&j)
    })
}

fn select_neighbors<'a>(
    neighbors: &'a Neighbors,
    is_acceptable_successor: impl Fn(&Vertice) -> bool + 'a,
) -> impl Fn(&Vertice) -> Vec<(Vertice, Cost)> + 'a {
    move |&(t, i): &Vertice| {
        neighbors
            .get(&(t, i))
            .expect("neighbors should have been pre-cached")
            .iter()
            .copied()
            .filter(|(v, _)| is_acceptable_successor(v))
            .collect()
    }
}
