//! Utilities to stream online algorithms.

use crate::algorithms::online::Memory;
use crate::algorithms::online::{OnlineAlgorithm, Step};
use crate::algorithms::Options;
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
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `next` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    /// * `options` - Algorithm options.
    pub fn stream<T, M, O>(
        &'a mut self,
        alg: impl OnlineAlgorithm<'a, T, P, M, O>,
        next: impl Fn(&mut Online<P>, &Schedule<T>) -> bool,
        options: O,
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
        options: O,
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
            let Step(x, m) = alg.next(self.clone(), xs, m, options.clone())?;
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
        options: O,
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
        options: O,
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
