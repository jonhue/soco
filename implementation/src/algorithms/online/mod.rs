//! Online Algorithms.

use crate::algorithms::Options;
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::{DefaultGivenOnlineProblem, Online, Problem};
use crate::result::Result;
use crate::schedule::Schedule;
use crate::value::Value;
use pyo3::{IntoPy, PyObject};
use serde::de::DeserializeOwned;
use serde::Serialize;

pub mod multi_dimensional;
pub mod uni_dimensional;

/// Solution fragment at some time $t$ to an online problem.
///
/// * Configuration at time $t$.
/// * Memory if new memory should be added.
pub struct Step<T, M>(pub Config<T>, pub Option<M>);
pub type IntegralStep<M> = Step<i32, M>;
pub type FractionalStep<M> = Step<f64, M>;

/// Memory of online algorithm.
pub trait Memory<'a, T, P, C, D>:
    Clone
    + std::fmt::Debug
    + DefaultGivenOnlineProblem<T, P, C, D>
    + DeserializeOwned
    + IntoPy<PyObject>
    + Send
    + Serialize
    + 'a
where
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}
impl<'a, T, P, C, D, M> Memory<'a, T, P, C, D> for M
where
    M: Clone
        + std::fmt::Debug
        + DefaultGivenOnlineProblem<T, P, C, D>
        + DeserializeOwned
        + IntoPy<PyObject>
        + Send
        + Serialize
        + 'a,
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}

/// Implementation of an online algorithm.
///
/// * `P` - problem
/// * `M` - memory
/// * `O` - options
///
/// Receives the arguments:
/// * `o` - online problem instance
/// * `t` - current time slot
/// * `xs` - schedule up to the previous time slot
/// * `prev_m` - latest memory, is the default if nothing was memorized
/// * `options` - algorithm options
pub trait OnlineAlgorithm<'a, T, P, M, O, C, D>:
    Fn(Online<P>, i32, &Schedule<T>, M, O) -> Result<Step<T, M>> + Send + Sync
where
    T: Value<'a>,
    P: Problem<T, C, D> + 'a,
    M: Memory<'a, T, P, C, D>,
    O: Options<T, P, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Executes the next iteration of an online algorithm.
    fn next(
        &self,
        o: Online<P>,
        xs: &Schedule<T>,
        prev_m_: Option<M>,
        options: O,
    ) -> Result<Step<T, M>> {
        let t = xs.t_end() + 1;
        let prev_m = match prev_m_ {
            None => M::default(&o),
            Some(prev_m) => prev_m,
        };

        self(o, t, xs, prev_m, options)
    }

    /// Executes the next iteration of an online algorithm with default options.
    fn next_with_default_options(
        &self,
        o: Online<P>,
        xs: &Schedule<T>,
        prev_m_: Option<M>,
    ) -> Result<Step<T, M>> {
        let options = O::default(&o.p);
        self.next(o, xs, prev_m_, options)
    }
}
impl<'a, T, P, M, O, C, D, F> OnlineAlgorithm<'a, T, P, M, O, C, D> for F
where
    T: Value<'a>,
    P: Problem<T, C, D> + 'a,
    M: Memory<'a, T, P, C, D>,
    O: Options<T, P, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
    F: Fn(Online<P>, i32, &Schedule<T>, M, O) -> Result<Step<T, M>>
        + Send
        + Sync,
{
}
