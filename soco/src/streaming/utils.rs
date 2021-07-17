//! Utilities to stream online algorithms.

use crate::algorithms::online::Memory;
use crate::algorithms::online::{OnlineAlgorithm, Step};
use crate::algorithms::Options;
use crate::config::Config;
use crate::problem::{Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use crate::value::Value;

impl<'a, P> Online<P>
where
    P: Problem + 'a,
{
    /// Utility to stream an online algorithm from `T = 1`.
    ///
    /// Returns resulting schedule, final memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `update` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    /// * `options` - Algorithm options.
    pub fn stream<T, M, O>(
        &mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
        update: impl Fn(&mut Online<P>, &Schedule<T>) -> bool,
        options: O,
    ) -> Result<(Schedule<T>, Option<M>)>
    where
        T: Value<'a>,
        M: Memory<'a, P>,
        O: Options<P>,
    {
        let mut xs = Schedule::empty();
        let m = self.stream_from(alg, update, options, &mut xs, None)?;
        Ok((xs, m))
    }

    /// Stream an online algorithm from an arbitrary initial time, given the previous schedule and memory.
    ///
    /// Returns final memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `update` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    /// * `options` - Algorithm options.
    /// * `xs` - Schedule.
    /// * `prev_m` - Memory of last iteration.
    pub fn stream_from<T, M, O>(
        &mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
        update: impl Fn(&mut Online<P>, &Schedule<T>) -> bool,
        options: O,
        xs: &mut Schedule<T>,
        mut prev_m: Option<M>,
    ) -> Result<Option<M>>
    where
        T: Value<'a>,
        M: Memory<'a, P>,
        O: Options<P>,
    {
        loop {
            prev_m = self.next(alg, options.clone(), xs, prev_m)?.1;
            if !update(self, &xs) {
                break;
            };
            self.verify()?;
        }

        Ok(prev_m)
    }

    /// Executes one step of an online algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `options` - Algorithm options.
    /// * `xs` - Schedule.
    /// * `prev_m` - Memory of last iteration.
    pub fn next<T, M, O>(
        &self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
        options: O,
        xs: &mut Schedule<T>,
        prev_m: Option<M>,
    ) -> Result<(Config<T>, Option<M>)>
    where
        T: Value<'a>,
        M: Memory<'a, P>,
        O: Options<P>,
    {
        assert(
            xs.t_end() + 1 == self.p.t_end(),
            Failure::OnlineInconsistentCurrentTimeSlot {
                previous_time_slots: xs.t_end(),
                current_time_slot: self.p.t_end(),
            },
        )?;

        let Step(x, m) = alg.next(self.clone(), xs, prev_m, options)?;
        xs.push(x.clone());
        Ok((x, m))
    }

    /// Utility to stream an online algorithm with a constant cost function from `T = 1`.
    ///
    /// Returns resulting schedule, final memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    pub fn offline_stream<T, M, O>(
        &mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
        t_end: i32,
        options: O,
    ) -> Result<(Schedule<T>, Option<M>)>
    where
        T: Value<'a>,
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
    /// Returns final memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    /// * `xs` - Schedule.
    /// * `prev_m` - Memory of last iteration.
    pub fn offline_stream_from<T, M, O>(
        &'a mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
        t_end: i32,
        options: O,
        xs: &mut Schedule<T>,
        prev_m: Option<M>,
    ) -> Result<Option<M>>
    where
        T: Value<'a>,
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
            prev_m,
        )
    }
}
