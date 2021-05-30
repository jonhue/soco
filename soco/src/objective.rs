//! Objective function.

use num::{NumCast, ToPrimitive};

use crate::config::Config;
use crate::problem::{
    SimplifiedSmoothedConvexOptimization, SmoothedBalancedLoadOptimization,
    SmoothedConvexOptimization, SmoothedLoadOptimization,
};
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::pos;
use crate::value::Value;

pub trait Objective<T>
where
    T: Value,
{
    /// Objective Function. Calculates the cost of a schedule.
    fn objective_function(&self, xs: &Schedule<T>) -> Result<f64> {
        let default = self.default_config();
        self.objective_function_with_default(xs, &default, false)
    }

    /// Inverted Objective Function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    fn inverted_objective_function(&self, xs: &Schedule<T>) -> Result<f64> {
        let default = self.default_config();
        self.objective_function_with_default(xs, &default, true)
    }

    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64>;

    fn default_config(&self) -> Config<T>;
}

impl<'a, T> Objective<T> for SmoothedConvexOptimization<'a, T>
where
    T: Value,
{
    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64> {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 2).unwrap_or(default);
            let x = &xs[t as usize - 1];
            cost += (self.hitting_cost)(t as i32, x.clone())
                .ok_or(Error::CostFnMustBeTotal)?;
            let delta = movement(x, prev_x, inverted);
            cost += (self.switching_cost)(delta);
        }
        Ok(cost)
    }

    fn default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

impl<'a, T> Objective<T> for SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value,
{
    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64> {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 2).unwrap_or(default);
            let x = &xs[t as usize - 1];
            cost += (self.hitting_cost)(t as i32, x.clone())
                .ok_or(Error::CostFnMustBeTotal)?;
            for k in 0..self.d as usize {
                let delta = ToPrimitive::to_f64(&scalar_movement(
                    x[k], prev_x[k], inverted,
                ))
                .unwrap();
                cost += self.switching_cost[k] * delta;
            }
        }
        Ok(cost)
    }

    fn default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

impl<T> Objective<T> for SmoothedLoadOptimization<T>
where
    T: Value,
{
    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64> {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 2).unwrap_or(default);
            let x = &xs[t as usize - 1];
            for k in 0..self.d as usize {
                cost +=
                    self.hitting_cost[k] * ToPrimitive::to_f64(&x[k]).unwrap();
                let delta = ToPrimitive::to_f64(&scalar_movement(
                    x[k], prev_x[k], inverted,
                ))
                .unwrap();
                cost += self.switching_cost[k] * delta;
            }
        }
        Ok(cost)
    }

    fn default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

impl<'a, T> Objective<T> for SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value,
{
    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64> {
        let ssco_p = self.to_ssco();
        ssco_p.objective_function_with_default(xs, default, inverted)
    }

    fn default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

pub fn scalar_movement<T>(j: T, prev_j: T, inverted: bool) -> T
where
    T: Value,
{
    pos(if inverted { prev_j - j } else { j - prev_j })
}

pub fn movement<T>(
    x: &Config<T>,
    prev_x: &Config<T>,
    inverted: bool,
) -> Config<T>
where
    T: Value,
{
    let mut result = Config::empty();
    for i in 0..x.d() as usize {
        result.push(scalar_movement(x[i], prev_x[i], inverted));
    }
    result
}
