//! Functions to convert between problem instances.

use crate::config::{Config, FractionalConfig, IntegralConfig};
use crate::cost::{CostFn, SingleCostFn};
use crate::norm::manhattan_scaled;
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization, Online,
    SimplifiedSmoothedConvexOptimization, SmoothedBalancedLoadOptimization,
    SmoothedConvexOptimization, SmoothedLoadOptimization,
};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::shift_time;
use crate::value::Value;
use noisy_float::prelude::*;
use num::{NumCast, ToPrimitive};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub trait DiscretizableVector {
    /// Ceil all elements of a vector.
    fn ceil(&self) -> Vec<i32>;
    /// Floor all elements of a vector.
    fn floor(&self) -> Vec<i32>;
}

impl<'a, T> DiscretizableVector for Vec<T>
where
    T: Value<'a>,
{
    fn ceil(&self) -> Vec<i32> {
        self.par_iter().map(|&x| x.ceil()).collect()
    }

    fn floor(&self) -> Vec<i32> {
        self.par_iter().map(|&x| x.floor()).collect()
    }
}

pub trait CastableVector<T> {
    fn to(&self) -> Vec<T>;
}

impl<'a, T, U> CastableVector<T> for Vec<U>
where
    T: Value<'a>,
    U: Value<'a>,
{
    fn to(&self) -> Vec<T> {
        self.par_iter()
            .map(|&x| NumCast::from(x).unwrap())
            .collect()
    }
}

pub trait DiscretizableCostFn<'a> {
    /// Discretize a certain fractional cost function.
    fn to_i(&'a self) -> CostFn<'a, IntegralConfig>;
}

impl<'a> DiscretizableCostFn<'a> for CostFn<'a, FractionalConfig> {
    fn to_i(&'a self) -> CostFn<'a, IntegralConfig> {
        CostFn::new(
            1,
            SingleCostFn::certain(move |t, x: IntegralConfig| {
                self.call_unbounded(t, x.to())
            }),
        )
    }
}

pub trait RelaxableCostFn<'a> {
    /// Relax a certain integral cost function to the fractional setting.
    fn to_f(&'a self) -> CostFn<'a, FractionalConfig>;
}

impl<'a> RelaxableCostFn<'a> for CostFn<'a, IntegralConfig> {
    fn to_f(&'a self) -> CostFn<'a, FractionalConfig> {
        CostFn::new(
            1,
            SingleCostFn::certain(move |t, x: FractionalConfig| {
                assert!(x.d() == 1, "cannot relax multidimensional problems");

                let j = n64(x[0]);
                if j.fract() == 0. {
                    self.call_unbounded(t, Config::single(j.to_i32().unwrap()))
                } else {
                    let l = self.call_unbounded(
                        t,
                        Config::single(j.floor().to_i32().unwrap()),
                    );
                    let u = self.call_unbounded(
                        t,
                        Config::single(j.ceil().to_i32().unwrap()),
                    );
                    (j.ceil() - j) * l + (j - j.floor()) * u
                }
            }),
        )
    }
}

pub trait DiscretizableProblem<'a> {
    type Output;

    /// Discretize a fractional problem instance.
    fn to_i(&'a self) -> Self::Output;
}

impl<'a> DiscretizableProblem<'a>
    for FractionalSimplifiedSmoothedConvexOptimization<'a>
{
    type Output = IntegralSimplifiedSmoothedConvexOptimization<'a>;

    fn to_i(&'a self) -> IntegralSimplifiedSmoothedConvexOptimization<'a> {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.floor(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.to_i(),
        }
    }
}

pub trait RelaxableProblem<'a> {
    type Output;

    /// Relax an integral problem instance to the fractional setting.
    fn to_f(&'a self) -> Self::Output;
}

impl<'a> RelaxableProblem<'a>
    for IntegralSimplifiedSmoothedConvexOptimization<'a>
{
    type Output = FractionalSimplifiedSmoothedConvexOptimization<'a>;

    fn to_f(&'a self) -> FractionalSimplifiedSmoothedConvexOptimization<'a> {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.to(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.to_f(),
        }
    }
}

impl<'a, T> SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    /// Convert to an instance of Smoothed Convex Optimization.
    /// This assumes that time slots are added after this conversion.
    pub fn into_sco(self) -> SmoothedConvexOptimization<'a, T> {
        let bounds = self
            .bounds
            .par_iter()
            .map(|&u| (NumCast::from(0).unwrap(), u))
            .collect();
        let switching_cost = manhattan_scaled(self.switching_cost.clone());
        SmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds,
            switching_cost: switching_cost.clone(),
            hitting_cost: CostFn::new(
                1,
                SingleCostFn::certain(move |t: i32, x: Config<T>| {
                    if t == self.t_end {
                        self.clone().hit_cost(t, x.clone()) + switching_cost(x)
                    } else {
                        self.clone().hit_cost(t, x)
                    }
                }),
            ),
        }
    }
}

impl<'a, T> SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value<'a>,
{
    /// Convert to an instance of Simplified Smoothed Convex Optimization.
    pub fn into_ssco(self) -> SimplifiedSmoothedConvexOptimization<'a, T> {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: CostFn::new(
                1,
                SingleCostFn::certain(move |t, x| self.clone().hit_cost(t, x)),
            ),
        }
    }
}

impl<'a, T> SmoothedLoadOptimization<T>
where
    T: Value<'a>,
{
    /// Convert instance to an instance of Smoothed Balanced-Load Optimization.
    pub fn into_sblo(self) -> SmoothedBalancedLoadOptimization<'a, T> {
        let hitting_cost = self
            .hitting_cost
            .par_iter()
            .map(|&c| CostFn::new(1, SingleCostFn::certain(move |_, _| n64(c))))
            .collect();
        SmoothedBalancedLoadOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost,
            load: self.load.clone(),
        }
    }
}

impl<'a, T> Online<T>
where
    T: DiscretizableProblem<'a>,
{
    /// Discretize online problem.
    pub fn to_i(&'a self) -> Online<T::Output> {
        Online {
            w: self.w,
            p: self.p.to_i(),
        }
    }
}

impl<'a, T> Online<T>
where
    T: RelaxableProblem<'a>,
{
    /// Relax online problem.
    pub fn to_f(&'a self) -> Online<T::Output> {
        Online {
            w: self.w,
            p: self.p.to_f(),
        }
    }
}

pub trait DiscretizableConfig {
    /// Ceil all elements of a config.
    fn ceil(&self) -> IntegralConfig;
    /// Floor all elements of a config.
    fn floor(&self) -> IntegralConfig;
}

impl<'a, T> DiscretizableConfig for Config<T>
where
    T: Value<'a>,
{
    fn ceil(&self) -> IntegralConfig {
        Config::new(self.to_vec().ceil())
    }

    fn floor(&self) -> IntegralConfig {
        Config::new(self.to_vec().floor())
    }
}

pub trait CastableConfig<T> {
    fn to(&self) -> Config<T>;
}

impl<'a, T, U> CastableConfig<T> for Config<U>
where
    T: Value<'a>,
    U: Value<'a>,
{
    fn to(&self) -> Config<T> {
        Config::new(self.to_vec().to())
    }
}

pub trait DiscretizableSchedule {
    /// Discretize a schedule.
    fn to_i(&self) -> IntegralSchedule;
}

impl<'a, T> DiscretizableSchedule for Schedule<T>
where
    T: Value<'a>,
{
    fn to_i(&self) -> IntegralSchedule {
        self.par_iter().map(|x| x.ceil()).collect()
    }
}

pub trait CastableSchedule<T> {
    fn to(&self) -> Schedule<T>;
}

impl<'a, T, U> CastableSchedule<T> for Schedule<U>
where
    T: Value<'a>,
    U: Value<'a>,
{
    fn to(&self) -> Schedule<T> {
        self.par_iter().map(|x| x.to()).collect()
    }
}

pub trait ResettableCostFn<'a, T> {
    /// Shift a certain cost function to some new initial time `t_start` (a time _before_ first time slot).
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T>;
}

impl<'a, T> ResettableCostFn<'a, T> for CostFn<'a, T>
where
    T: Clone,
{
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T> {
        CostFn::new(
            1,
            SingleCostFn::certain(move |t, j| {
                self.call_unbounded(shift_time(t, t_start + 1), j)
            }),
        )
    }
}

pub trait ResettableProblem<'a, T> {
    /// Shifts problem instance to some new initial time `t_start`.
    fn reset(&'a self, t_start: i32) -> Self;
}

impl<'a, T> ResettableProblem<'a, T>
    for SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    fn reset(
        &'a self,
        t_start: i32,
    ) -> SimplifiedSmoothedConvexOptimization<'a, T> {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end - t_start,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.reset(t_start),
        }
    }
}
