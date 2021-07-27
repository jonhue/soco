//! Objective function.

use crate::config::Config;
use crate::cost::{Cost, RawCost};
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::Problem;
use crate::result::Result;
use crate::schedule::Schedule;
use crate::utils::pos;
use crate::value::Value;
use noisy_float::prelude::*;
use num::NumCast;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait Objective<'a, T, C, D>: Problem<T, C, D> + Sync
where
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Objective Function. Calculates the cost of a schedule.
    fn objective_function(&self, xs: &Schedule<T>) -> Result<Cost<C, D>> {
        let default = self._default_config();
        self._objective_function_with_default(xs, &default, 1., false)
    }

    /// Inverted Objective Function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    fn inverted_objective_function(
        &self,
        xs: &Schedule<T>,
    ) -> Result<Cost<C, D>> {
        let default = self._default_config();
        self._objective_function_with_default(xs, &default, 1., true)
    }

    /// `\alpha`-unfair Objective Function. Calculates the cost of a schedule.
    fn alpha_unfair_objective_function(
        &self,
        xs: &Schedule<T>,
        alpha: f64,
    ) -> Result<Cost<C, D>> {
        let default = self._default_config();
        self._objective_function_with_default(xs, &default, alpha, false)
    }

    fn objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
    ) -> Result<Cost<C, D>> {
        self._objective_function_with_default(xs, default, 1., false)
    }

    fn _objective_function_with_default(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        alpha: f64,
        inverted: bool,
    ) -> Result<Cost<C, D>> {
        Ok(sum_over_schedule(
            self.t_end(),
            xs,
            default,
            |t, prev_x, x| {
                let hitting_cost = self.hit_cost(t as i32, x.clone());
                Cost::new(
                    hitting_cost.cost
                        + n64(alpha) * self.movement(prev_x, x, inverted),
                    hitting_cost.output,
                )
            },
        ))
    }

    /// Movement in the decision space.
    fn total_movement(&self, xs: &Schedule<T>, inverted: bool) -> Result<N64> {
        let default = self._default_config();
        Ok(
            sum_over_schedule(self.t_end(), xs, &default, |_, prev_x, x| {
                RawCost::raw(self.movement(prev_x, x, inverted))
            })
            .cost,
        )
    }

    fn _default_config(&self) -> Config<T> {
        Config::repeat(NumCast::from(0).unwrap(), self.d())
    }
}
impl<'a, T, C, D, P> Objective<'a, T, C, D> for P
where
    P: Problem<T, C, D> + Sync,
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}

fn sum_over_schedule<'a, T, C, D>(
    t_end: i32,
    xs: &Schedule<T>,
    default: &Config<T>,
    f: impl Fn(i32, Config<T>, Config<T>) -> Cost<C, D> + Send + Sync,
) -> Cost<C, D>
where
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
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
