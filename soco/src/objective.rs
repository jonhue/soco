//! Objective function.

use num::{Num, NumCast, ToPrimitive};

use crate::config::Config;
use crate::problem::{
    SmoothedBalancedLoadOptimization, SmoothedConvexOptimization,
    SmoothedLoadOptimization,
};
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::pos;

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
        let default_x = Config::<T>::repeat(NumCast::from(0).unwrap(), self.d);
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 2).unwrap_or(&default_x);
            let x = &xs[t as usize - 1];
            cost += (self.hitting_cost)(t as i32, x.to_vec())
                .ok_or(Error::CostFnMustBeTotal)?;
            for k in 0..self.d as usize {
                let delta = movement(x[k], prev_x[k], inverted);
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
        let default_x = Config::<T>::repeat(NumCast::from(0).unwrap(), self.d);
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 2).unwrap_or(&default_x);
            let x = &xs[t as usize - 1];
            for k in 0..self.d as usize {
                cost +=
                    self.hitting_cost[k] * ToPrimitive::to_f64(&x[k]).unwrap();
                let delta = movement(x[k], prev_x[k], inverted);
                cost += self.switching_cost[k] * delta;
            }
        }
        Ok(cost)
    }
}

impl<'a, T> Objective<T> for SmoothedBalancedLoadOptimization<'a, T>
where
    T: Copy + Num + NumCast + PartialOrd,
{
    fn _objective_function(
        &self,
        xs: &Schedule<T>,
        inverted: bool,
    ) -> Result<f64> {
        let sco_p = self.to_sco();
        if inverted {
            sco_p.inverted_objective_function(xs)
        } else {
            sco_p.objective_function(xs)
        }
    }
}

pub fn movement<T>(x: T, prev_x: T, inverted: bool) -> f64
where
    T: Num + NumCast + PartialOrd,
{
    ToPrimitive::to_f64(&pos(if inverted { prev_x - x } else { x - prev_x }))
        .unwrap()
}
