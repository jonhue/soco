//! Algorithms.

use crate::{
    model::{ModelOutputFailure, ModelOutputSuccess},
    problem::{DefaultGivenProblem, Problem},
};

mod capacity_provisioning;

pub mod offline;
pub mod online;

/// Options of algorithm.
pub trait Options<T, P, C, D>:
    Clone + DefaultGivenProblem<T, P, C, D> + Send
where
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}
impl<'a, T, P, C, D, O> Options<T, P, C, D> for O
where
    O: Clone + DefaultGivenProblem<T, P, C, D> + Send,
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}
