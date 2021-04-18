use crate::problem::types::DiscreteHomProblem;
use crate::problem::utils::project;

impl<'a> DiscreteHomProblem<'a> {
    pub fn ilcp(&self, i: i32) -> i32 {
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
