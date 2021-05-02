use std::collections::HashMap;
use std::f64::INFINITY;
use std::sync::Arc;

use crate::problem::{DiscreteHomProblem, HomProblem};
use crate::result::{Error, Result};
use crate::schedule::DiscreteSchedule;
use crate::utils::{assert, is_pow_of_2, pos};

/// The minimal cost from some initial vertice alongside the shortest path to the final vertice.
type Path = (DiscreteSchedule, f64);
/// Maps a vertice to its minimal cost from some initial vertice alongside the shortest path.
type Paths = HashMap<(i32, i32), Path>;

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

    let k_init = if p.m > 2 {
        (p.m as f64).log(2.).floor() as u32 - 2
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

fn select_initial_rows<'a>(
    p: &'a DiscreteHomProblem<'a>,
) -> impl Fn(i32) -> Vec<i32> + 'a {
    move |_| (0..=4).map(|e| e * p.m / 4).collect()
}

fn select_next_rows<'a>(
    p: &'a DiscreteHomProblem<'a>,
    xs: &'a DiscreteSchedule,
    k: u32,
) -> impl Fn(i32) -> Vec<i32> + 'a {
    move |t| {
        (-2..=2)
            .map(|e| xs[t as usize - 1] + e * 2_i32.pow(k))
            .filter(|&j| 0 <= j && j <= p.m)
            .collect()
    }
}

fn find_schedule(
    p: &DiscreteHomProblem<'_>,
    select_rows: impl Fn(i32) -> Vec<i32>,
    inverted: bool,
) -> Result<(DiscreteSchedule, f64)> {
    let mut paths: Paths = HashMap::new();
    paths.insert((0, 0), (vec![], 0.));

    let mut prev_rows = vec![0];
    for t in 1..=p.t_end {
        let rows = select_rows(t);

        for &j in &rows {
            for &i in &prev_rows {
                let c = build_cost(p, t, i, j, inverted)?;
                match paths.get(&(t, j)) {
                    None => {
                        update_paths(&mut paths, t, i, j, c)?;
                    }
                    Some(&(_, prev_c)) => {
                        if c < prev_c {
                            update_paths(&mut paths, t, i, j, c)?;
                        };
                    }
                };
            }
        }

        prev_rows = rows;
    }

    let mut result = &(vec![], INFINITY);
    for i in prev_rows {
        let path =
            paths.get(&(p.t_end, i)).ok_or(Error::PathsShouldBeCached)?;
        if path.1 < result.1 {
            result = path;
        }
    }

    Ok(result.clone())
}

fn build_cost(
    p: &DiscreteHomProblem<'_>,
    t: i32,
    i: i32,
    j: i32,
    inverted: bool,
) -> Result<f64> {
    Ok(p.beta * pos(if inverted { i - j } else { j - i }) as f64
        + (p.f)(t, j).ok_or(Error::CostFnMustBeTotal)?)
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
    let xs = [&prev_xs[..], &[j]].concat();

    paths.insert(v, (xs, c));
    Ok(())
}
