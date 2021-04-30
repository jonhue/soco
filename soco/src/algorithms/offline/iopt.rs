use ordered_float::OrderedFloat;
use pathfinding::directed::dijkstra::dijkstra;
use std::collections::HashMap;
use std::sync::Arc;

use crate::problem::{DiscreteHomProblem, HomProblem};
use crate::result::{Error, Result};
use crate::schedule::DiscreteSchedule;
use crate::utils::{assert, ipos, is_pow_of_2};

/// Represents a vertice `v_{t, j}` where the `t ~ time` and `j ~ #servers`.
type Vertice = (i32, i32);
/// Represents the length (cost) of an edge.
type Cost = OrderedFloat<f64>;
/// Maps a vertice to all its neighbors with some cost.
type Neighbors = HashMap<Vertice, Vec<(Vertice, Cost)>>;

impl<'a> DiscreteHomProblem<'a> {
    /// Discrete Deterministic Offline Algorithm
    pub fn iopt(&'a self) -> Result<(DiscreteSchedule, Cost)> {
        assert(is_pow_of_2(self.m), Error::MustBePowOf2)?;

        let neighbors = self.build_neighbors()?;

        let k_init = if self.m > 2 {
            (self.m as f64).log(2.).floor() as u32 - 2
        } else {
            0
        };

        let initial_neighbors = self.select_initial_neighbors(&neighbors);
        let mut result = self.find_schedule(initial_neighbors);

        if k_init > 0 {
            for k in k_init - 1..=0 {
                let next_neighbors =
                    self.select_next_neighbors(&result.0, &neighbors, k);
                result = self.find_schedule(next_neighbors);
            }
        }

        Ok(result)
    }

    /// Utility to transform a problem instance where `m` is not a power of `2` to an instance that is accepted by `iopt`.
    pub fn transform(&'a self) -> DiscreteHomProblem<'a> {
        let m = 2_i32.pow((self.m as f64).log(2.).ceil() as u32);
        let f = Arc::new(move |t, x| {
            if x <= self.m {
                (self.f)(t, x)
            } else {
                Some(
                    x as f64
                        * ((self.f)(t, self.m).unwrap() + std::f64::EPSILON),
                )
            }
        });

        HomProblem {
            m,
            t_end: self.t_end,
            beta: self.beta,
            f,
        }
    }

    fn build_neighbors(&self) -> Result<Neighbors> {
        let mut neighbors = HashMap::new();
        neighbors.insert((0, 0), self.build_edges(0, 0)?);
        for t in 1..=self.t_end {
            for i in 0..self.m {
                neighbors.insert((t, i), self.build_edges(t, i)?);
            }
        }
        Ok(neighbors)
    }

    fn build_edges(&self, t: i32, i: i32) -> Result<Vec<(Vertice, Cost)>> {
        if t == self.t_end {
            Ok(vec![((self.t_end + 1, 0), OrderedFloat(0.))])
        } else {
            vec![0; self.m as usize]
                .iter()
                .enumerate()
                .map(|(j, _)| {
                    Ok((
                        (t + 1, j as i32),
                        self.build_cost(t + 1, i, j as i32)?,
                    ))
                })
                .collect()
        }
    }

    fn build_cost(&self, t: i32, i: i32, j: i32) -> Result<Cost> {
        Ok(OrderedFloat(
            self.beta * ipos(j - i) as f64
                + (self.f)(t, j).ok_or(Error::CostFnMustBeTotal)?,
        ))
    }

    fn find_schedule(
        &self,
        neighbors: impl Fn(&Vertice) -> Vec<(Vertice, Cost)> + 'a,
    ) -> (DiscreteSchedule, Cost) {
        let result = dijkstra(&(0, 0), neighbors, |&(t, j): &Vertice| {
            (t, j) == (self.t_end + 1, 0)
        });
        let (mut xs, cost) = result.expect("there should always be a path");
        xs.remove(0);
        xs.remove(xs.len() - 1);
        (xs.into_iter().map(|(_, j)| j).collect(), cost)
    }

    fn select_initial_neighbors(
        &self,
        neighbors: &'a Neighbors,
    ) -> impl Fn(&Vertice) -> Vec<(Vertice, Cost)> + 'a {
        let acceptable_successors: Vec<i32> =
            (0..=4).map(|e| e * self.m / 4).collect();
        select_neighbors(neighbors, move |&(_, j)| {
            acceptable_successors.contains(&j)
        })
    }

    fn select_next_neighbors(
        &'a self,
        xs: &DiscreteSchedule,
        neighbors: &'a Neighbors,
        k: u32,
    ) -> impl Fn(&Vertice) -> Vec<(Vertice, Cost)> + 'a {
        let acceptable_successors: Vec<Vec<i32>> = (1..=self.t_end)
            .map(|t| {
                (-2..=2)
                    .map(|e| xs[t as usize - 1] + e * 2_i32.pow(k))
                    .collect()
            })
            .collect();
        select_neighbors(neighbors, move |&(t, j)| {
            t == self.t_end + 1
                || acceptable_successors[t as usize - 1].contains(&j)
        })
    }
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
