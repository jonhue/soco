//! Helper functions for simulating online problems.

use crate::problem::{HomProblem, Online, OnlineSolution, Schedule};

impl<'a, T> Online<HomProblem<'a, T>>
where
    T: Copy,
{
    /// Utility to stream an online algorithm.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `U` - Memory.
    /// * `alg` - Online algorithm to stream.
    /// * `next` - Callback that in each iteration updates the problem instance. Return `true` to continue stream, `false` to end stream.
    pub fn stream<U>(
        &mut self,
        alg: impl Fn(
            &Online<HomProblem<'a, T>>,
            &Schedule<T>,
            &Vec<U>,
        ) -> OnlineSolution<T, U>,
        next: impl Fn(&mut Online<HomProblem<'a, T>>, &Schedule<T>, &Vec<U>) -> bool,
    ) -> (Schedule<T>, Vec<U>) {
        let mut xs = vec![];
        let mut ms = vec![];

        loop {
            assert!(
                self.p.t_end == self.w + xs.len() as i32 + 1,
                "online problem must contain precisely the information for the next iteration"
            );

            let (i, m) = alg(self, &xs, &ms);
            xs.push(i);
            ms.push(m);
            if !next(self, &xs, &ms) {
                break;
            };
        }

        (xs, ms)
    }

    /// Utility to stream an online algorithm with a constant cost function.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `U` - Memory.
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    pub fn offline_stream<U>(
        &mut self,
        alg: impl Fn(
            &Online<HomProblem<'a, T>>,
            &Schedule<T>,
            &Vec<U>,
        ) -> OnlineSolution<T, U>,
        t_end: i32,
    ) -> (Schedule<T>, Vec<U>) {
        self.stream(alg, |o, _, _| {
            if o.p.t_end < t_end as i32 {
                o.p.t_end = o.p.t_end + 1;
                true
            } else {
                false
            }
        })
    }
}
