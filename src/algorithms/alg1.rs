use ordered_float::OrderedFloat;
use pathfinding::dijkstra;

use crate::lib::types::{DiscreteHomProblem, DiscreteSchedule, HomProblem};
use crate::lib::utils::discrete_pos;

// Represents a vertice `v_{t, j}` where the `t ~ time` and `j ~ #servers`.
type Vertice = (i32, i32);
// Represents the length (cost) of an edge
type Cost = OrderedFloat<f64>;
// Maps a vertice to all its neighbors with some cost.
type Neighbors = Box<dyn Fn(&Vertice) -> Vec<(Vertice, Cost)>>;

static eps: f64 = 1.;

pub fn alg1(mut p: &'static DiscreteHomProblem) -> DiscreteSchedule {
    if (p.m as f64).log(2.) % 1. == 0. {
        p = &transform_problem(p);
    }

    let neighbors = build_neighbors(p);

    let initial_neighbors = select_initial_neighbors(p, neighbors);
    let mut xs = find_schedule(p, initial_neighbors);

    let k_init = (p.m as f64).log(2.).floor() as u32 - 3;
    for k in k_init..0 {
        let next_neighbors = select_next_neighbors(p, &xs, neighbors, k);
        xs = find_schedule(p, next_neighbors);
    }
    return xs;
}

fn transform_problem(p: &'static DiscreteHomProblem) -> DiscreteHomProblem {
    let m = (2 as i32).pow((p.m as f64).log(2.).ceil() as u32);
    let f = Box::new(move |t, x| {
        if x <= p.m {
            (p.f)(t, x)
        } else {
            Some(
                x as f64
                    * ((p.f)(t, p.m).expect("f should be total on its domain")
                        + eps),
            )
        }
    });

    return HomProblem {
        m: m,
        t_end: p.t_end,
        beta: p.beta,
        f: f,
    };
}

fn build_neighbors(p: &'static DiscreteHomProblem) -> Neighbors {
    return Box::new(move |&(t, i): &Vertice| {
        if t == p.t_end {
            vec![((t + 1, 0), OrderedFloat(0.))]
        } else {
            vec![0; p.m as usize]
                .iter()
                .enumerate()
                .map(|(j, _)| {
                    ((t + 1, j as i32), build_cost(p, t, i, j as i32))
                })
                .collect()
        }
    });
}

fn build_cost(p: &DiscreteHomProblem, t: i32, i: i32, j: i32) -> Cost {
    return OrderedFloat(
        p.beta * discrete_pos(i - j)
            + (p.f)(t, i).expect("f should be total on its domain"),
    );
}

fn select_initial_neighbors(
    p: &DiscreteHomProblem,
    neighbors: Neighbors,
) -> Neighbors {
    let acceptable_successors: Vec<i32> = (0..4).map(|e| e * p.m / 4).collect();
    return select_neighbors(
        neighbors,
        Box::new(move |&(_, j)| acceptable_successors.contains(&j)),
    );
}

fn select_next_neighbors(
    p: &DiscreteHomProblem,
    xs: &DiscreteSchedule,
    neighbors: Neighbors,
    k: u32,
) -> Neighbors {
    let acceptable_successors: Vec<Vec<i32>> = (1..p.t_end)
        .map(|t| {
            (-2..2)
                .map(|e| xs[t as usize - 1] + e * (2 as i32).pow(k))
                .collect()
        })
        .collect();
    return select_neighbors(
        neighbors,
        Box::new(move |&(t, j)| acceptable_successors[t as usize - 1].contains(&j)),
    );
}

fn select_neighbors(
    neighbors: Neighbors,
    is_acceptable_successor: Box<dyn Fn(&Vertice) -> bool>,
) -> Neighbors {
    return Box::new(move |&(t, i): &Vertice| {
        neighbors(&(t, i))
            .iter()
            .map(|&x| x) // TODO: why is this necessary?
            .filter(|&(v, _)| is_acceptable_successor(&v))
            .collect()
    });
}

fn find_schedule(
    p: &DiscreteHomProblem,
    neighbors: Neighbors,
) -> DiscreteSchedule {
    let result = dijkstra(&(0, 0), neighbors, |&(t, j): &Vertice| {
        t == p.t_end && j == 0
    });
    let (xs, _) = result.expect("there should always be a path");
    return xs.into_iter().map(|(_, j)| j).collect();
}
