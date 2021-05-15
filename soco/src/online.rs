//! Online problems.

use crate::config::Config;
use crate::problem::Problem;
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use crate::value::Value;

/// Online instance of a problem.
pub struct Online<T> {
    /// Problem.
    pub p: T,
    /// Finite, non-negative prediction window.
    ///
    /// This prediction window is included in the time bound of the problem instance.
    pub w: i32,
}

/// Solution fragment at some time `t` to an online problem.
///
/// * Configuration at time `t`.
/// * Memory if new memory should be added.
pub struct Step<T, U>(pub Config<T>, pub Option<U>)
where
    T: Value;

impl<'a, T> Online<T>
where
    T: Problem,
{
    /// Utility to stream an online algorithm from `T = 1`.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `next` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    /// * `options` - Algorithm options.
    pub fn stream<U, V, W>(
        &mut self,
        alg: impl Fn(
            &Online<T>,
            &mut Schedule<V>,
            &mut Vec<U>,
            &W,
        ) -> Result<Step<V, U>>,
        next: impl Fn(&mut Online<T>, &Schedule<V>, &Vec<U>) -> bool,
        options: &W,
    ) -> Result<(Schedule<V>, Vec<U>)>
    where
        V: Value,
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
    pub fn stream_from<U, V, W>(
        &mut self,
        alg: impl Fn(
            &Online<T>,
            &mut Schedule<V>,
            &mut Vec<U>,
            &W,
        ) -> Result<Step<V, U>>,
        next: impl Fn(&mut Online<T>, &Schedule<V>, &Vec<U>) -> bool,
        options: &W,
        xs: &mut Schedule<V>,
        ms: &mut Vec<U>,
    ) -> Result<()>
    where
        V: Value,
    {
        assert(
            xs.t_end() as usize == ms.len(),
            Error::OnlineInsufficientInformation,
        )?;

        loop {
            let t = xs.t_end() + 1;
            assert(
                self.p.t_end() == t + self.w,
                Error::OnlineInsufficientInformation,
            )?;

            let Step(x, m) = alg(self, xs, ms, options)?;
            xs.push(x);
            match m {
                None => (),
                Some(m) => ms.push(m),
            };
            if !next(self, &xs, &ms) {
                break;
            };
        }

        Ok(())
    }

    /// Utility to stream an online algorithm with a constant cost function from `T = 1`.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `U` - Memory.
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    pub fn offline_stream<U, V, W>(
        &mut self,
        alg: impl Fn(
            &Online<T>,
            &mut Schedule<V>,
            &mut Vec<U>,
            &W,
        ) -> Result<Step<V, U>>,
        t_end: i32,
        options: &W,
    ) -> Result<(Schedule<V>, Vec<U>)>
    where
        V: Value,
    {
        self.stream(
            alg,
            |o, _, _| {
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
    /// * `U` - Memory.
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    /// * `xs` - Schedule.
    /// * `ms` - Memory.
    pub fn offline_stream_from<U, V, W>(
        &mut self,
        alg: impl Fn(
            &Online<T>,
            &mut Schedule<V>,
            &mut Vec<U>,
            &W,
        ) -> Result<Step<V, U>>,
        t_end: i32,
        options: &W,
        xs: &mut Schedule<V>,
        ms: &mut Vec<U>,
    ) -> Result<()>
    where
        V: Value,
    {
        self.stream_from(
            alg,
            |o, _, _| {
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
