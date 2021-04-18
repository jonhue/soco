use crate::problem::types::{DiscreteHomProblem, DiscreteSchedule, Online};
use crate::problem::utils::project;

impl<'a> Online<DiscreteHomProblem<'a>> {
    pub fn lcp(&self, xs: DiscreteSchedule) -> i32 {
        let i = if xs.len() > 0 { xs[xs.len() - 1] } else { 0 };
        let l = self.lower_bound();
        let u = self.upper_bound();
        project(i, l, u)
    }

    fn lower_bound(&self) -> i32 {
        1
    }

    fn upper_bound(&self) -> i32 {
        1
    }
}
