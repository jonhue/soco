//! Objective function.

use crate::config::{Config, IntegralConfig};
use crate::convert::{CastableConfig, CastableSchedule};
use crate::problem::{
    Problem, SimplifiedSmoothedConvexOptimization, SmoothedConvexOptimization,
    SmoothedLoadOptimization,
};
use crate::result::{Failure, Result};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::{assert, pos};
use crate::value::Value;
use noisy_float::prelude::*;
use num::NumCast;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait Objective<'a, T>
where
    T: Value<'a>,
{
    /// Objective Function. Calculates the cost of a schedule.
    fn objective_function(&self, xs: &Schedule<T>) -> Result<N64> {
        let default = self._default_config();
        self._objective_function_with_default(xs, &default, 1., false)
    }

    /// Inverted Objective Function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    fn inverted_objective_function(&self, xs: &Schedule<T>) -> Result<N64> {
        let default = self._default_config();
        self._objective_function_with_default(xs, &default, 1., true)
    }

    /// `\alpha`-unfair Objective Function. Calculates the cost of a schedule.
    fn alpha_unfair_objective_function(
        &self,
        xs: &Schedule<T>,
        alpha: f64,
    ) -> Result<N64> {
        let default = self._default_config();
        self._objective_function_with_default(xs, &default, alpha, false)
    }

    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
    ) -> Result<N64> {
        self._objective_function_with_default(xs, default, 1., false)
    }

    fn _objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        alpha: f64,
        inverted: bool,
    ) -> Result<N64>;

    /// Movement in the decision space.
    fn movement(&self, xs: &Schedule<T>, inverted: bool) -> Result<N64> {
        let default = self._default_config();
        self._movement_with_default(xs, &default, inverted)
    }

    fn _movement_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<N64>;

    fn _default_config(&self) -> Config<T>;
}

impl<'a, T> Objective<'a, T> for SmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    fn _objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        alpha: f64,
        inverted: bool,
    ) -> Result<N64> {
        assert(!inverted, Failure::UnsupportedInvertedCost)?;

        Ok(sum_over_schedule(
            self.t_end,
            xs,
            default,
            |t, prev_x, x| {
                self.hit_cost(t as i32, x.clone())
                    + n64(alpha) * (self.switching_cost)(x - prev_x)
            },
        ))
    }

    fn _movement_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<N64> {
        assert(!inverted, Failure::UnsupportedInvertedCost)?;

        Ok(sum_over_schedule(
            self.t_end,
            xs,
            default,
            |_, prev_x, x| (self.switching_cost)(x - prev_x),
        ))
    }

    fn _default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

impl<'a, T> Objective<'a, T> for SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    fn _objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        alpha: f64,
        inverted: bool,
    ) -> Result<N64> {
        Ok(sum_over_schedule(
            self.t_end,
            xs,
            default,
            |t, prev_x, x| {
                self.hit_cost(t as i32, x.clone())
                    + (0..self.d as usize)
                        .into_iter()
                        .map(|k| {
                            let delta: N64 = NumCast::from(scalar_movement(
                                x[k], prev_x[k], inverted,
                            ))
                            .unwrap();
                            n64(alpha) * n64(self.switching_cost[k]) * delta
                        })
                        .sum::<N64>()
            },
        ))
    }

    fn _movement_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<N64> {
        Ok(sum_over_schedule(
            self.t_end,
            xs,
            default,
            |_, prev_x, x| {
                (0..self.d as usize)
                    .into_iter()
                    .map(|k| -> N64 {
                        let delta: N64 = NumCast::from(scalar_movement(
                            x[k], prev_x[k], inverted,
                        ))
                        .unwrap();
                        n64(self.switching_cost[k]) * delta
                    })
                    .sum()
            },
        ))
    }

    fn _default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

/// Implements integral objective for a relaxed problem instance which also implements a fractional objective.
/// Used by the uni-dimensional randomized algorithm.
impl<'a, P> Objective<'a, i32> for P
where
    P: Problem + Objective<'a, f64>,
{
    fn _objective_function_with_default(
        &self,
        xs: &IntegralSchedule,
        default: &IntegralConfig,
        alpha: f64,
        inverted: bool,
    ) -> Result<N64> {
        self._objective_function_with_default(
            &xs.to(),
            &default.to(),
            alpha,
            inverted,
        )
    }

    fn _movement_with_default(
        &self,
        xs: &IntegralSchedule,
        default: &IntegralConfig,
        inverted: bool,
    ) -> Result<N64> {
        self._movement_with_default(&xs.to(), &default.to(), inverted)
    }

    fn _default_config(&self) -> IntegralConfig {
        Config::repeat(0, self.d())
    }
}

impl<'a, T> Objective<'a, T> for SmoothedLoadOptimization<T>
where
    T: Value<'a>,
{
    fn _objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        alpha: f64,
        inverted: bool,
    ) -> Result<N64> {
        Ok(sum_over_schedule(
            self.t_end,
            xs,
            default,
            |_, prev_x, x| {
                (0..self.d as usize)
                    .into_iter()
                    .map(|k| -> N64 {
                        let delta: N64 = NumCast::from(scalar_movement(
                            x[k], prev_x[k], inverted,
                        ))
                        .unwrap();
                        let j: N64 = NumCast::from(x[k]).unwrap();
                        n64(self.hitting_cost[k]) * j
                            + n64(alpha) * n64(self.switching_cost[k]) * delta
                    })
                    .sum()
            },
        ))
    }

    fn _movement_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        inverted: bool,
    ) -> Result<N64> {
        Ok(sum_over_schedule(
            self.t_end,
            xs,
            default,
            |_, prev_x, x| {
                (0..self.d as usize)
                    .into_iter()
                    .map(|k| -> N64 {
                        let delta: N64 = NumCast::from(scalar_movement(
                            x[k], prev_x[k], inverted,
                        ))
                        .unwrap();
                        n64(self.switching_cost[k]) * delta
                    })
                    .sum()
            },
        ))
    }

    fn _default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d)
    }
}

fn sum_over_schedule<'a, T>(
    t_end: i32,
    xs: &Schedule<T>,
    default: &Config<T>,
    f: impl Fn(i32, Config<T>, Config<T>) -> N64 + Send + Sync,
) -> N64
where
    T: Value<'a>,
{
    (1..=t_end)
        .into_par_iter()
        .map(|t| {
            let prev_x = xs.get(t - 1).unwrap_or(default).clone();
            let x = xs.get(t).unwrap().clone();
            f(t, prev_x, x)
        })
        .sum()
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
    (0..x.d() as usize)
        .into_iter()
        .map(|k| scalar_movement(x[k], prev_x[k], inverted))
        .collect()
}
