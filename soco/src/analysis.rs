//! Analysis functions.

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, DiscreteHomProblem,
    DiscreteSchedule,
};
use crate::utils::{faccess, fpos, iaccess, ipos};

impl<'a> ContinuousHomProblem<'a> {
    /// Objective Function. Calculates the cost of a schedule.
    pub fn objective_function(&self, xs: &ContinuousSchedule) -> f64 {
        self._objective_function(xs, false)
    }

    /// Inverted Objective Function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    pub fn inverted_objective_function(&self, xs: &ContinuousSchedule) -> f64 {
        self._objective_function(xs, false)
    }

    fn _objective_function(
        &self,
        xs: &ContinuousSchedule,
        inverted: bool,
    ) -> f64 {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = faccess(xs, t - 2);
            let x = faccess(xs, t - 1);
            cost += (self.f)(t, x).unwrap()
                + self.beta
                    * fpos(if inverted { prev_x - x } else { x - prev_x })
                        as f64;
        }
        cost
    }
}

impl<'a> DiscreteHomProblem<'a> {
    /// Objective Function. Calculates the cost of a schedule.
    pub fn objective_function(&self, xs: &DiscreteSchedule) -> f64 {
        self._objective_function(xs, false)
    }

    /// Inverted Objective Function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    pub fn inverted_objective_function(&self, xs: &DiscreteSchedule) -> f64 {
        self._objective_function(xs, false)
    }

    fn _objective_function(
        &self,
        xs: &DiscreteSchedule,
        inverted: bool,
    ) -> f64 {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = iaccess(xs, t - 2);
            let x = iaccess(xs, t - 1);
            cost += (self.f)(t, x).unwrap()
                + self.beta
                    * ipos(if inverted { prev_x - x } else { x - prev_x })
                        as f64;
        }
        cost
    }
}