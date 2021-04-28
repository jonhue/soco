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
                o.p.t_end == o.w + xs.len() as i32 + 1,
                "online problem must contain precisely the information for the next iteration"
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

    /// Utility to stream an online algorithm with a predefined cost function.
    ///
    /// Returns resulting schedule, memory of the algorithm.
    ///
    /// * `U` - Memory.
    /// * `alg` - Online algorithm to stream.
    /// * `t_end` - Finite time horizon.
    pub fn offline_stream<U>(
        &self,
        alg: impl Fn(
            &Online<HomProblem<'a, T>>,
            &Schedule<T>,
            &Vec<U>,
        ) -> OnlineSolution<T, U>,
        t_end: i32,
    ) -> (Schedule<T>, Vec<U>) {
        self.stream(alg, |o, xs, _| {
            if xs.len() < t_end as usize {
                Some(Online {
                    w: o.w,
                    p: HomProblem {
                        m: o.p.m,
                        t_end: o.p.t_end + 1,
                        f: Box::new(|_, _| Some(1.)), // TODO: copy f
                        beta: o.p.beta,
                    },
                })
            } else {
                None
            }
        })
    }
}
