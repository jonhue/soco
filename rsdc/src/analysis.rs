//! Analysis functions.

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, DiscreteHomProblem,
    DiscreteSchedule,
};
use crate::utils::{fpos, ipos};

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
        for t in 1..=self.t_end + 1 {
            let prev_x = if t > 1 && t as usize <= xs.len() + 1 {
                xs[t as usize - 2]
            } else {
                0.
            };
            let x = if t as usize <= xs.len() {
                xs[t as usize - 1]
            } else {
                0.
            };
            cost += (self.f)(t, x).expect("f should be total on its domain")
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
        for t in 1..=self.t_end + 1 {
            let prev_x = if t > 1 && t as usize <= xs.len() + 1 {
                xs[t as usize - 2]
            } else {
                0
            };
            let x = if t as usize <= xs.len() {
                xs[t as usize - 1]
            } else {
                0
            };
            cost += (self.f)(t, x).expect("f should be total on its domain")
                + self.beta
                    * ipos(if inverted { prev_x - x } else { x - prev_x })
                        as f64;
        }
        cost
    }
}
