//! Lazy Capacity Provisioning

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, Online, OnlineSolution,
};
use crate::utils::fproject;

/// Lower and upper bound at some time t.
type Memory<T> = (T, T);

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// (Continuous) Extended Lazy Capacity Provisioning
    pub fn elcp(
        &self,
        xs: &ContinuousSchedule,
        _: &Vec<Memory<f64>>,
    ) -> OnlineSolution<f64, Memory<f64>> {
        assert_eq!(self.w, 0);

        let i = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };
        let l = self.lower_bound();
        let u = self.upper_bound();
        let j = fproject(i, l, u);
        (j, (l, u))
    }

    fn lower_bound(&self) -> f64 {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &ContinuousHomProblem<'a>|
             -> f64 { p.objective_function(&xs.to_vec()) };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1]
    }

    fn upper_bound(&self) -> f64 {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &ContinuousHomProblem<'a>|
             -> f64 { p.inverted_objective_function(&xs.to_vec()) };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1]
    }
}
