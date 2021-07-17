//! Online Algorithms.

use crate::algorithms::Options;
use crate::config::Config;
use crate::problem::{DefaultGivenProblem, Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use crate::value::Value;
use serde::{Deserialize, Serialize};

pub mod multi_dimensional;
pub mod uni_dimensional;

/// Solution fragment at some time `t` to an online problem.
///
/// * Configuration at time `t`.
/// * Memory if new memory should be added.
pub struct Step<T, M>(pub Config<T>, pub Option<M>);
pub type IntegralStep<M> = Step<i32, M>;
pub type FractionalStep<M> = Step<f64, M>;

/// Memory of online algorithm.
pub trait Memory<'a, P>:
    Clone + DefaultGivenProblem<P> + Deserialize<'a> + Serialize + 'a
where
    P: Problem,
{
}
impl<'a, T, P> Memory<'a, P> for T
where
    T: Clone + DefaultGivenProblem<P> + Deserialize<'a> + Serialize + 'a,
    P: Problem,
{
}

/// Implementation of an online algorithm.
///
/// * `T` - Value (integral, fractional).
/// * `P` - Problem.
/// * `M` - Memory.
/// * `O` - Options.
///
/// Receives the arguments:
/// * `o` - Online problem instance.
/// * `t` - Current time slot.
/// * `xs` - Schedule up to the previous time slot.
/// * `prev_m` - Latest memory, is the default if nothing was memorized.
/// * `options` - Algorithm options.
pub trait OnlineAlgorithm<'a, T, P, M, O>:
    Fn(Online<P>, i32, &Schedule<T>, M, O) -> Result<Step<T, M>>
where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
{
    fn next(
        &self,
        o: Online<P>,
        xs: &Schedule<T>,
        prev_m_: Option<M>,
        options: O,
    ) -> Result<Step<T, M>> {
        let t = xs.t_end() + 1;
        assert(
            o.p.t_end() == t + o.w,
            Failure::OnlineInsufficientInformation {
                t_end: o.p.t_end(),
                t,
                w: o.w,
            },
        )?;
        let prev_m = match prev_m_ {
            None => M::default(&o.p),
            Some(prev_m) => prev_m,
        };

        self(o, t, xs, prev_m, options)
    }
}
impl<'a, T, P, M, O, F> OnlineAlgorithm<'a, T, P, M, O> for F
where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
    F: Fn(Online<P>, i32, &Schedule<T>, M, O) -> Result<Step<T, M>>,
{
}

pub trait OnlineAlgorithmWithDefaultOptions<'a, T, P, M, O>:
    OnlineAlgorithm<'a, T, P, M, O>
where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
{
    fn next_with_default_options(
        &self,
        o: Online<P>,
        xs: &Schedule<T>,
        prev_m_: Option<M>,
    ) -> Result<Step<T, M>> {
        let options = O::default(&o.p);
        OnlineAlgorithm::next(self, o, xs, prev_m_, options)
    }
}
impl<'a, T, P, M, O, F> OnlineAlgorithmWithDefaultOptions<'a, T, P, M, O> for F
where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
    F: Fn(Online<P>, i32, &Schedule<T>, M, O) -> Result<Step<T, M>>,
{
}
