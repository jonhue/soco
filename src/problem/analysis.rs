//! Analysis functions.

use crate::problem::types::{DiscreteHomProblem, DiscreteSchedule};
use crate::problem::utils::ipos;

impl<'a> DiscreteHomProblem<'a> {
    /// Objective Function. Calculates the cost of a schedule.
    pub fn objective_function(&self, xs: &DiscreteSchedule) -> f64 {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = if t > 1 { xs[t as usize - 2] } else { 0 };
            let x = xs[t as usize - 1];
            cost += (self.f)(t, x).expect("f should be total on its domain")
                + self.beta * ipos(x - prev_x) as f64;
        }
        cost
    }
}
