//! Online problems.

use crate::config::Config;
use crate::problem::Problem;
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::assert;

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
/// * Memory.
pub struct OnlineSolution<T, U>(pub Config<T>, pub U);

impl<'a, T> Online<T>
where
    T: Problem,
{
    /// Utility to stream an online algorithm.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `alg` - Online algorithm to stream.
    /// * `next` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    pub fn stream<U, V>(
        &mut self,
        alg: impl Fn(
            &Online<T>,
            &Schedule<V>,
            &Vec<U>,
        ) -> Result<OnlineSolution<V, U>>,
        next: impl Fn(&mut Online<T>, &Schedule<V>, &Vec<U>) -> bool,
    ) -> Result<(Schedule<V>, Vec<U>)>
    where
        V: Clone,
    {
        let mut xs = Schedule::empty();
        let mut ms = vec![];

        loop {
            let t = xs.t_end() + 1;
            assert(
                self.p.t_end() == t + self.w,
                Error::OnlineInsufficientInformation,
            )?;

            let OnlineSolution(i, m) = alg(self, &xs, &ms)?;
            xs.push(i);
            ms.push(m);
            if !next(self, &xs, &ms) {
                break;
            };
        }

        Ok((xs, ms))
    }

    /// Utility to stream an online algorithm with a constant cost function.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `U` - Memory.
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    pub fn offline_stream<U, V>(
        &mut self,
        alg: impl Fn(
            &Online<T>,
            &Schedule<V>,
            &Vec<U>,
        ) -> Result<OnlineSolution<V, U>>,
        t_end: i32,
    ) -> Result<(Schedule<V>, Vec<U>)>
    where
        V: Clone,
    {
        self.stream(alg, |o, _, _| {
            if o.p.t_end() < t_end - o.w {
                o.p.inc_t_end();
                true
            } else {
                false
            }
        })
    }
}
