use ordered_float::OrderedFloat;
use pathfinding::dijkstra;

#[path = "lib.rs"]
mod lib;
use lib::types::{DiscreteProblem, DiscreteSchedule};
use lib::utils::log;

// Represents a vertice `v_{t, j}` where the `t ~ time` and `j ~ #servers`.
type Vertice = (usize, i32);
// Represents the length (weight) of an edge
type Weight = OrderedFloat<f64>;
// Maps a vertice to all its neighbors with some weight.
type Neighbors = fn(&Vertice) -> Vec<(Vertice, Weight)>;

pub fn alg1(problem: &DiscreteProblem) -> DiscreteSchedule {
    let neighbors = |&(t, j): &Vertice| {
        vec![(t, j)]
            .into_iter()
            .map(|(t, j)| ((t, j), OrderedFloat(1.)))
            .collect()
    };

    let initial_neighbors = select_initial_neighbors(&problem, neighbors);
    let schedule = find_schedule(&problem, initial_neighbors);
    let k_init = log(problem.m) - 3;
    for k in k_init..0 {
        let next_neighbors = select_next_neighbors(&problem, &schedule, neighbors, k);
        schedule = find_schedule(&problem, next_neighbors);
    }
    return schedule;
}

fn select_initial_neighbors(problem: &DiscreteProblem, neighbors: Neighbors) -> Neighbors {}

fn select_next_neighbors(
    problem: &DiscreteProblem,
    schedule: &DiscreteSchedule,
    neighbors: Neighbors,
    k: i32,
) -> Neighbors {
}

fn find_schedule(problem: &DiscreteProblem, neighbors: Neighbors) -> DiscreteSchedule {
    let result = dijkstra(&(0, 0), neighbors, |&(t, j): &Vertice| {
        t == problem.t_end && j == 0
    });
    let (schedule, _) = result.expect("There should always be a path.");
    return schedule.into_iter().map(|(t, j)| j).collect();
}
