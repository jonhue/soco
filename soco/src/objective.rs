//! Objective function.

use crate::config::{Config, IntegralConfig};
use crate::convert::{RelaxableConfig, RelaxableSchedule};
use crate::problem::{
    Problem, SimplifiedSmoothedConvexOptimization, SmoothedConvexOptimization,
    SmoothedLoadOptimization,
};
use crate::result::{Failure, Result};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::{assert, pos};
use crate::value::Value;
use num::{NumCast, ToPrimitive};

pub trait Objective<'a, T>
where
    T: Value<'a>,
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

impl<'a, T> Objective<'a, T> for SmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64> {
        assert(!inverted, Failure::UnsupportedInvertedCost)?;

        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 1).unwrap_or(default).clone();
            let x = xs.get(t).unwrap().clone();
            cost += self.hit_cost(t as i32, x.clone()).raw();
            cost += (self.switching_cost)(x - prev_x).raw();
        }
        Ok(cost)
    }

    fn default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

impl<'a, T> Objective<'a, T> for SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64> {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 1).unwrap_or(default);
            let x = xs.get(t).unwrap();
            cost += self.hit_cost(t as i32, x.clone()).raw();
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

/// Implements integral objective for a relaxed problem instance which also implements a fractional objective.
/// Used by the uni-dimensional randomized algorithm.
impl<'a, P> Objective<'a, i32> for P
where
    P: Problem + Objective<'a, f64>,
{
    fn objective_function_with_default(
        &self,
        xs: &IntegralSchedule,
        default: &IntegralConfig,
        inverted: bool,
    ) -> Result<f64> {
        self.objective_function_with_default(
            &xs.to_f(),
            &default.to_f(),
            inverted,
        )
    }

    fn default_config(&self) -> IntegralConfig {
        Config::repeat(0, self.d())
    }
}

impl<'a, T> Objective<'a, T> for SmoothedLoadOptimization<T>
where
    T: Value<'a>,
{
    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<f64> {
        let mut cost = 0.;
        for t in 1..=self.t_end {
            let prev_x = xs.get(t - 1).unwrap_or(default);
            let x = xs.get(t).unwrap();
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

pub fn scalar_movement<'a, T>(j: T, prev_j: T, inverted: bool) -> T
where
    T: Value<'a>,
{
    pos(if inverted { prev_j - j } else { j - prev_j })
}

pub fn movement<'a, T>(
    x: &Config<T>,
    prev_x: &Config<T>,
    inverted: bool,
) -> Config<T>
where
    T: Value<'a>,
{
    let mut result = Config::empty();
    for i in 0..x.d() as usize {
        result.push(scalar_movement(x[i], prev_x[i], inverted));
    }
    result
}
