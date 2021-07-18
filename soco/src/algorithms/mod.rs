//! Algorithms.

use crate::problem::{DefaultGivenProblem, Problem};

mod capacity_provisioning;
mod graph_search;

pub mod offline;
pub mod online;

/// Options of algorithm.
pub trait Options<P>: Clone + DefaultGivenProblem<P>
where
    P: Problem,
{
}
impl<T, P> Options<P> for T
where
    T: Clone + DefaultGivenProblem<P>,
    P: Problem,
{
}
