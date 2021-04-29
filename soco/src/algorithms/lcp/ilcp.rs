//! Lazy Capacity Provisioning

use crate::problem::{
    DiscreteHomProblem, DiscreteSchedule, Online, OnlineSolution,
};
use crate::schedule::DiscretizableSchedule;
use crate::utils::iproject;

/// Lower and upper bound at some time t.
type Memory<T> = (T, T);

impl<'a> Online<DiscreteHomProblem<'a>> {
    /// Integer Lazy Capacity Provisioning
    pub fn ilcp(
        &self,
        xs: &DiscreteSchedule,
        _: &Vec<Memory<i32>>,
    ) -> OnlineSolution<i32, Memory<i32>> {
        assert_eq!(self.w, 0);

        let i = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
        let l = self.lower_bound();
        let u = self.upper_bound();
        let j = iproject(i, l, u);
        (j, (l, u))
    }

    fn lower_bound(&self) -> i32 {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &DiscreteHomProblem<'a>|
             -> f64 { p.objective_function(&xs.to_vec().to_i()) };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1].ceil() as i32
    }

    fn upper_bound(&self) -> i32 {
        let objective_function = |xs: &[f64],
                                  _: Option<&mut [f64]>,
                                  p: &mut &DiscreteHomProblem<'a>|
         -> f64 {
            p.inverted_objective_function(&xs.to_vec().to_i())
        };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1].ceil() as i32
    }
}
