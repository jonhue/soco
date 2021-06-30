//! Online problems.

use crate::config::Config;
use crate::problem::Problem;
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use crate::value::Value;

/// Online instance of a problem.
#[derive(Clone)]
pub struct Online<T> {
    /// Problem.
    pub p: T,
    /// Finite, non-negative prediction window.
    ///
    /// This prediction window is included in the time bound of the problem instance,
    /// i.e. at time `t` `t_end` should be set to `t + w`.
    pub w: i32,
}

/// Solution fragment at some time `t` to an online problem.
///
/// * Configuration at time `t`.
/// * Memory if new memory should be added.
pub struct Step<T, M>(pub Config<T>, pub Option<M>)
where
    T: Value;
pub type IntegralStep<M> = Step<i32, M>;
pub type FractionalStep<M> = Step<f64, M>;

/// Gives a type default value which may depend on a problem instance.
pub trait DefaultGivenProblem<P>
where
    P: Problem,
{
    fn default(p: &P) -> Self;
}
impl<T, P> DefaultGivenProblem<P> for T
where
    T: Default,
    P: Problem,
{
    fn default(_: &P) -> Self {
        T::default()
    }
}

/// Memory of online algorithm.
pub trait Memory<'a, P>: Clone + DefaultGivenProblem<P> + 'a
where
    P: Problem,
{
}
impl<'a, T, P> Memory<'a, P> for T
where
    T: Clone + DefaultGivenProblem<P> + 'a,
    P: Problem,
{
}

/// Options of online algorithm.
pub trait Options<P>: DefaultGivenProblem<P>
where
    P: Problem,
{
}
impl<T, P> Options<P> for T
where
    T: DefaultGivenProblem<P>,
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
    Fn(Online<P>, i32, &Schedule<T>, M, &O) -> Result<Step<T, M>>
where
    T: Value,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
{
    fn next(
        &self,
        o: Online<P>,
        xs: &Schedule<T>,
        prev_m_: Option<M>,
        options: &O,
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
    T: Value,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
    F: Fn(Online<P>, i32, &Schedule<T>, M, &O) -> Result<Step<T, M>>,
{
}

impl<'a, P> Online<P>
where
    P: Problem + 'a,
{
    /// Utility to stream an online algorithm from `T = 1`.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `next` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    /// * `options` - Algorithm options.
    pub fn stream<T, M, O>(
        &'a mut self,
        alg: impl OnlineAlgorithm<'a, T, P, M, O>,
        next: impl Fn(&mut Online<P>, &Schedule<T>) -> bool,
        options: &O,
    ) -> Result<(Schedule<T>, Vec<M>)>
    where
        T: Value,
        M: Memory<'a, P>,
        O: Options<P>,
    {
        let mut xs = Schedule::empty();
        let mut ms = vec![];

        self.stream_from(alg, next, options, &mut xs, &mut ms)?;
        Ok((xs, ms))
    }

    /// Stream an online algorithm from an arbitrary initial time, given the previous schedule and memory.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `next` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    /// * `options` - Algorithm options.
    /// * `xs` - Schedule.
    /// * `ms` - Memory.
    pub fn stream_from<T, M, O>(
        &'a mut self,
        alg: impl OnlineAlgorithm<'a, T, P, M, O>,
        next: impl Fn(&mut Online<P>, &Schedule<T>) -> bool,
        options: &O,
        xs: &mut Schedule<T>,
        ms: &mut Vec<M>,
    ) -> Result<()>
    where
        T: Value,
        M: Memory<'a, P>,
        O: Options<P>,
    {
        assert(
            xs.t_end() as usize == ms.len(),
            Failure::OnlineOutOfDateMemory {
                previous_time_slots: xs.t_end(),
                memory_entries: ms.len() as i32,
            },
        )?;

        loop {
            let m = if !ms.is_empty() {
                ms.get(ms.len() - 1).cloned()
            } else {
                None
            };
            let Step(x, m) = alg.next(self.clone(), xs, m, options)?;
            xs.push(x);
            match m {
                None => (),
                Some(m) => ms.push(m),
            };
            if !next(self, &xs) {
                break;
            };
        }

        Ok(())
    }

    /// Utility to stream an online algorithm with a constant cost function from `T = 1`.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    pub fn offline_stream<T, M, O>(
        &'a mut self,
        alg: impl OnlineAlgorithm<'a, T, P, M, O>,
        t_end: i32,
        options: &O,
    ) -> Result<(Schedule<T>, Vec<M>)>
    where
        T: Value,
        M: Memory<'a, P>,
        O: Options<P>,
    {
        self.stream(
            alg,
            |o, _| {
                if o.p.t_end() < t_end - o.w {
                    o.p.inc_t_end();
                    true
                } else {
                    false
                }
            },
            options,
        )
    }

    /// Utility to stream an online algorithm with a constant cost function
    /// from an arbitrary initial time, given the previous schedule and memory.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    /// * `xs` - Schedule.
    /// * `ms` - Memory.
    pub fn offline_stream_from<T, M, O>(
        &'a mut self,
        alg: impl OnlineAlgorithm<'a, T, P, M, O>,
        t_end: i32,
        options: &O,
        xs: &mut Schedule<T>,
        ms: &'a mut Vec<M>,
    ) -> Result<()>
    where
        T: Value,
        M: Memory<'a, P>,
        O: Options<P>,
    {
        self.stream_from(
            alg,
            |o, _| {
                if o.p.t_end() < t_end - o.w {
                    o.p.inc_t_end();
                    true
                } else {
                    false
                }
            },
            options,
            xs,
            ms,
        )
    }
}
