//! Utilities to stream online algorithms.

use crate::algorithms::online::Memory;
use crate::algorithms::online::{OnlineAlgorithm, Step};
use crate::algorithms::Options;
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::{Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use crate::value::Value;

impl<'a, P> Online<P> {
    /// Utility to stream an online algorithm from `T = 1`.
    ///
    /// Returns resulting schedule, final memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `update` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    /// * `options` - Algorithm options.
    pub fn stream<T, M, O, C, D>(
        &mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O, C, D>,
        update: impl Fn(&mut Online<P>, &Schedule<T>) -> bool,
        options: O,
    ) -> Result<(Schedule<T>, Option<M>)>
    where
        T: Value<'a>,
        P: Problem<T, C, D> + 'a,
        M: Memory<'a, T, P, C, D>,
        O: Options<T, P, C, D>,
        C: ModelOutputSuccess,
        D: ModelOutputFailure,
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
    pub fn stream_from<T, M, O, C, D>(
        &mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O, C, D>,
        update: impl Fn(&mut Online<P>, &Schedule<T>) -> bool,
        options: O,
        xs: &mut Schedule<T>,
        mut prev_m: Option<M>,
    ) -> Result<Option<M>>
    where
        T: Value<'a>,
        P: Problem<T, C, D> + 'a,
        M: Memory<'a, T, P, C, D>,
        O: Options<T, P, C, D>,
        C: ModelOutputSuccess,
        D: ModelOutputFailure,
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
    pub fn next<T, M, O, C, D>(
        &self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O, C, D>,
        options: O,
        xs: &mut Schedule<T>,
        prev_m: Option<M>,
    ) -> Result<(Config<T>, Option<M>)>
    where
        T: Value<'a>,
        P: Problem<T, C, D> + 'a,
        M: Memory<'a, T, P, C, D>,
        O: Options<T, P, C, D>,
        C: ModelOutputSuccess,
        D: ModelOutputFailure,
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
    /// * `t_end` - Finite time horizon. `t_end >= 1`.
    pub fn offline_stream<T, M, O, C, D>(
        &mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O, C, D>,
        t_end: i32,
        options: O,
    ) -> Result<(Schedule<T>, Option<M>)>
    where
        T: Value<'a>,
        P: Problem<T, C, D> + 'a,
        M: Memory<'a, T, P, C, D>,
        O: Options<T, P, C, D>,
        C: ModelOutputSuccess,
        D: ModelOutputFailure,
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
    /// * `t_end` - Finite time horizon. Must be greater or equals to the current time slot.
    /// * `xs` - Schedule.
    /// * `prev_m` - Memory of last iteration.
    pub fn offline_stream_from<T, M, O, C, D>(
        &'a mut self,
        alg: &impl OnlineAlgorithm<'a, T, P, M, O, C, D>,
        t_end: i32,
        options: O,
        xs: &mut Schedule<T>,
        prev_m: Option<M>,
    ) -> Result<Option<M>>
    where
        T: Value<'a>,
        P: Problem<T, C, D> + 'a,
        M: Memory<'a, T, P, C, D>,
        O: Options<T, P, C, D>,
        C: ModelOutputSuccess,
        D: ModelOutputFailure,
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
