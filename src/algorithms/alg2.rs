//! Discrete Lazy Capacity Provisioning

use crate::problem::{
    DiscreteHomProblem, DiscreteSchedule, Online, OnlineSolution,
};
use crate::utils::project;

/// Lower and upper bound at some time t.
type Memory = (i32, i32);

impl<'a> Online<DiscreteHomProblem<'a>> {
    pub fn lcp(
        &self,
        xs: &DiscreteSchedule,
        _: &Vec<Memory>,
    ) -> OnlineSolution<i32, Memory> {
        assert_eq!(self.w, 0);

        let i = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
        let l = self.lower_bound();
        let u = self.upper_bound();
        (project(i, l, u), (l, u))
    }

    fn lower_bound(&self) -> i32 {
        1
    }

    fn upper_bound(&self) -> i32 {
        1
    }
}
