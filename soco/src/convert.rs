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
use crate::schedule::{FractionalSchedule, IntegralSchedule};
use crate::utils::{shift_time, unshift_time};
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use noisy_float::prelude::*;
use num::{NumCast, ToPrimitive};

pub trait DiscretizableVector {
    /// Ceil all elements of a vector.
    fn ceil(&self) -> Vec<i32>;
    /// Floor all elements of a vector.
    fn floor(&self) -> Vec<i32>;
}

impl DiscretizableVector for Vec<f64> {
    fn ceil(&self) -> Vec<i32> {
        self.iter().map(|&x| x.ceil() as i32).collect()
    }

    fn floor(&self) -> Vec<i32> {
        self.iter().map(|&x| x.floor() as i32).collect()
    }
}

pub trait RelaxableVector {
    /// Convert an integral vector to a fractional vector.
    fn to_f(&self) -> Vec<f64>;
}

impl RelaxableVector for Vec<i32> {
    fn to_f(&self) -> Vec<f64> {
        self.iter().map(|&x| x as f64).collect()
    }
}

pub trait DiscretizableCostFn<'a> {
    /// Discretize a fractional cost function.
    fn to_i(&'a self) -> CostFn<'a, IntegralConfig>;
}

impl<'a> DiscretizableCostFn<'a> for CostFn<'a, FractionalConfig> {
    fn to_i(&'a self) -> CostFn<'a, IntegralConfig> {
        CostFn::stretch(
            1,
            self.now(),
            SingleCostFn::predictive(move |t, x: IntegralConfig| {
                self.call_unbounded_predictive(t, x.to_f())
            }),
        )
    }
}

pub trait RelaxableCostFn<'a> {
    /// Relax an integral cost function to the fractional setting.
    fn to_f(&'a self) -> CostFn<'a, FractionalConfig>;
}

impl<'a> RelaxableCostFn<'a> for CostFn<'a, IntegralConfig> {
    fn to_f(&'a self) -> CostFn<'a, FractionalConfig> {
        CostFn::stretch(
            1,
            self.now(),
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
            bounds: self.bounds.to_f(),
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
    pub fn into_sco(self) -> SmoothedConvexOptimization<'a, T> {
        let bounds = self
            .bounds
            .iter()
            .map(|&u| (NumCast::from(0).unwrap(), u))
            .collect();
        SmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds,
            switching_cost: manhattan_scaled(self.switching_cost),
            hitting_cost: self.hitting_cost.clone(),
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
            .iter()
            .map(|&l| SingleCostFn::certain(move |_, _| n64(l)))
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
            hitting_cost: CostFn::stretch(
                1,
                self.t_end,
                SingleCostFn::certain(move |t, x| self.clone().hit_cost(t, x)),
            ),
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

impl DiscretizableConfig for FractionalConfig {
    fn ceil(&self) -> IntegralConfig {
        Config::new(self.to_vec().ceil())
    }

    fn floor(&self) -> IntegralConfig {
        Config::new(self.to_vec().floor())
    }
}

pub trait RelaxableConfig {
    /// Convert an integral config to a fractional config.
    fn to_f(&self) -> FractionalConfig;
}

impl RelaxableConfig for IntegralConfig {
    fn to_f(&self) -> FractionalConfig {
        Config::new(self.to_vec().to_f())
    }
}

pub trait DiscretizableSchedule {
    /// Discretize a schedule.
    fn to_i(&self) -> IntegralSchedule;
}

impl DiscretizableSchedule for FractionalSchedule {
    fn to_i(&self) -> IntegralSchedule {
        self.iter().map(|x| x.ceil()).collect()
    }
}

pub trait RelaxableSchedule {
    /// Relax an integral schedule to a fractional schedule.
    fn to_f(&self) -> FractionalSchedule;
}

impl RelaxableSchedule for IntegralSchedule {
    fn to_f(&self) -> FractionalSchedule {
        self.iter().map(|x| x.to_f()).collect()
    }
}

pub trait ResettableCostFn<'a, T> {
    /// Shift a cost function to some new initial time `t_start` (time _before_ first time slot).
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T>;
}

impl<'a, T> ResettableCostFn<'a, T> for CostFn<'a, T>
where
    T: Clone,
{
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T> {
        CostFn::stretch(
            1,
            unshift_time(self.now(), t_start + 1),
            SingleCostFn::predictive(move |t, j| {
                self.call_unbounded_predictive(shift_time(t, t_start + 1), j)
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
