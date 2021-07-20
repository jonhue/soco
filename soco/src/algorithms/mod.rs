//! Algorithms.

use crate::problem::{DefaultGivenProblem, Problem};

mod capacity_provisioning;

pub mod offline;
pub mod online;

/// Options of algorithm.
pub trait Options<P>: Clone + DefaultGivenProblem<P> + Send
where
    P: Problem,
{
}
impl<T, P> Options<P> for T
where
    T: Clone + DefaultGivenProblem<P> + Send,
    P: Problem,
{
}
