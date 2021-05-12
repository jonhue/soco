//! Objective function.

use num::{Num, NumCast, ToPrimitive};

use crate::problem::{SmoothedConvexOptimization, SmoothedLoadOptimization};
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

impl<'a, T> Objective<T> for SmoothedConvexOptimization<'a, T>
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
            let prev_x = access(
                xs,
                t - 2,
                vec![NumCast::from(0).unwrap(); self.d as usize],
            );
            let x = xs[t as usize - 1].clone();
            cost += (self.hitting_cost)(t as i32, x.clone())
                .ok_or(Error::CostFnMustBeTotal)?;
            for k in 0..self.d as usize {
                let delta = ToPrimitive::to_f64(&pos(if inverted {
                    prev_x[k] - x[k]
                } else {
                    x[k] - prev_x[k]
                }))
                .unwrap();
                cost += self.switching_cost[k] * delta;
            }
        }
        Ok(cost)
    }
}

impl<T> Objective<T> for SmoothedLoadOptimization<T>
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
            let prev_x = access(
                xs,
                t - 2,
                vec![NumCast::from(0).unwrap(); self.d as usize],
            );
            let x = &xs[t as usize - 1];
            for k in 0..self.d as usize {
                cost +=
                    self.hitting_cost[k] * ToPrimitive::to_f64(&x[k]).unwrap();
                let delta = ToPrimitive::to_f64(&pos(if inverted {
                    prev_x[k] - x[k]
                } else {
                    x[k] - prev_x[k]
                }))
                .unwrap();
                cost += self.switching_cost[k] * delta;
            }
        }
        Ok(cost)
    }
}
