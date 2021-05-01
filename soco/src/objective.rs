//! Objective function.

use num::{Num, NumCast, ToPrimitive};

use crate::problem::HomProblem;
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::{access, pos};

pub trait Objective<T> {
    /// Objective Function. Calculates the cost of a schedule.
    fn objective_function(&self, xs: &Schedule<T>) -> Result<f64> {
        self._objective_function(xs, false)
    }

    /// Inverted Objective Function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    fn inverted_objective_function(&self, xs: &Schedule<T>) -> Result<f64> {
        self._objective_function(xs, true)
    }

    fn _objective_function(
        &self,
        xs: &Schedule<T>,
        inverted: bool,
    ) -> Result<f64>;
}

impl<'a, T> Objective<T> for HomProblem<'a, T>
where
    T: Copy + Num + NumCast + PartialOrd,
{
    fn _objective_function(
        &self,
        xs: &Schedule<T>,
        inverted: bool,
    ) -> Result<f64> {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = access(xs, t - 2);
            let x = access(xs, t - 1);
            cost += (self.f)(t, x).ok_or(Error::CostFnMustBeTotal)?
                + self.beta
                    * ToPrimitive::to_f64(&pos(if inverted {
                        prev_x - x
                    } else {
                        x - prev_x
                    }))
                    .unwrap();
        }
        Ok(cost)
    }
}
