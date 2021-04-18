//! Helper functions for simulating online problems.

use crate::problem::types::{HomProblem, Online, OnlineSolution, Schedule};

impl<'a, T> Online<HomProblem<'a, T>>
where
    T: Copy,
{
    /// Utility to stream an online algorithm.
    ///
    /// Stores resulting schedule and memory of the algorithm.
    ///
    /// * `U` - Memory.
    /// * `alg` - Online algorithm to stream.
    /// * `next` - Callback that in each iteration returns the next problem instance. Return `None` to end stream.
    pub fn stream<U>(
        &self,
        alg: impl Fn(
            &Online<HomProblem<'a, T>>,
            &Schedule<T>,
            &Vec<U>,
        ) -> OnlineSolution<T, U>,
        next: impl Fn(
            &Online<HomProblem<'a, T>>,
            &Schedule<T>,
            &Vec<U>,
        ) -> Option<Online<HomProblem<'a, T>>>,
    ) -> (Schedule<T>, Vec<U>) {
        let mut xs = vec![];
        let mut ms = vec![];
        let mut o = self;

        let mut tmp;
        loop {
            assert!(
                o.p.t_end > o.w + xs.len() as i32,
                "online problem must contain information for the next iteration"
            );

            let (i, m) = alg(o, &xs, &ms);
            xs.push(i);
            ms.push(m);
            o = match next(o, &xs, &ms) {
                None => break,
                Some(o) => {
                    tmp = o;
                    &tmp
                }
            };
        }

        (xs, ms)
    }
}
