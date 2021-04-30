//! Analysis functions.

use crate::problem::HomProblem;
use crate::result::Result;
use crate::schedule::Schedule;

use private::Objective;

impl<'a, T> HomProblem<'a, T>
where
    HomProblem<'a, T>: Objective<T>,
{
    /// Objective Function. Calculates the cost of a schedule.
    pub fn objective_function(&self, xs: &Schedule<T>) -> Result<f64> {
        self._objective_function(xs, false)
    }

    /// Inverted Objective Function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    pub fn inverted_objective_function(&self, xs: &Schedule<T>) -> Result<f64> {
        self._objective_function(xs, true)
    }
}

mod private {
    use crate::problem::{ContinuousHomProblem, DiscreteHomProblem};
    use crate::result::{Error, Result};
    use crate::schedule::{ContinuousSchedule, DiscreteSchedule, Schedule};
    use crate::utils::{faccess, fpos, iaccess, ipos};

    pub trait Objective<T> {
        fn _objective_function(
            &self,
            xs: &Schedule<T>,
            inverted: bool,
        ) -> Result<f64>;
    }

    impl<'a> Objective<f64> for ContinuousHomProblem<'a> {
        fn _objective_function(
            &self,
            xs: &ContinuousSchedule,
            inverted: bool,
        ) -> Result<f64> {
            let mut cost = 0.;
            for t in 1..=self.t_end {
                let prev_x = faccess(xs, t - 2);
                let x = faccess(xs, t - 1);
                cost += (self.f)(t, x).ok_or(Error::CostFnMustBeTotal)?
                    + self.beta
                        * fpos(if inverted { prev_x - x } else { x - prev_x })
                            as f64;
            }
            Ok(cost)
        }
    }

    impl<'a> Objective<i32> for DiscreteHomProblem<'a> {
        fn _objective_function(
            &self,
            xs: &DiscreteSchedule,
            inverted: bool,
        ) -> Result<f64> {
            let mut cost = 0.;
            for t in 1..=self.t_end {
                let prev_x = iaccess(xs, t - 2);
                let x = iaccess(xs, t - 1);
                cost += (self.f)(t, x).ok_or(Error::CostFnMustBeTotal)?
                    + self.beta
                        * ipos(if inverted { prev_x - x } else { x - prev_x })
                            as f64;
            }
            Ok(cost)
        }
    }
}
