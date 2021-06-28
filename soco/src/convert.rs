//! Functions to convert between problem instances.

use crate::config::{Config, FractionalConfig, IntegralConfig};
use crate::cost::data_center::load::Load;
use crate::cost::data_center::{apply_loads, load_balance};
use crate::cost::CostFn;
use crate::norm::NormFn;
use crate::online::Online;
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization,
    SimplifiedSmoothedConvexOptimization, SmoothedBalancedLoadOptimization,
    SmoothedConvexOptimization, SmoothedLoadOptimization,
};
use crate::schedule::{FractionalSchedule, IntegralSchedule};
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use num::{NumCast, ToPrimitive};
use std::sync::Arc;

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
        Arc::new(move |t, x| self(t, x.to_f()))
    }
}

pub trait RelaxableCostFn<'a> {
    /// Relax an integral cost function to the fractional setting.
    fn to_f(&'a self) -> CostFn<'a, FractionalConfig>;
}

impl<'a> RelaxableCostFn<'a> for CostFn<'a, IntegralConfig> {
    fn to_f(&'a self) -> CostFn<'a, FractionalConfig> {
        Arc::new(move |t, x| {
            assert!(x.d() == 1, "cannot relax multidimensional problems");

            self(t, x.ceil())
        })
        // Arc::new(move |t, x| {
        //     assert!(x.d() == 1, "cannot relax multidimensional problems");

        //     let j = x[0];
        //     if j.fract() == 0. {
        //         self(t, Config::single(j as i32))
        //     } else {
        //         let l = self(t, Config::single(j.floor() as i32));
        //         let u = self(t, Config::single(j.ceil() as i32));
        //         if l.is_none() || u.is_none() {
        //             return None;
        //         }

        //         Some((j.ceil() - j) * l.unwrap() + (j - j.floor()) * u.unwrap())
        //     }
        // })
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
    T: Value,
{
    /// Convert to an instance of Smoothed Convex Optimization.
    pub fn to_sco(&'a self) -> SmoothedConvexOptimization<'a, T> {
        let bounds = self
            .bounds
            .iter()
            .map(|&u| (NumCast::from(0).unwrap(), u))
            .collect();
        let switching_cost: NormFn<'a, Config<T>> = Arc::new(move |x| {
            let mut result = 0.;
            for k in 0..self.d as usize {
                result += self.switching_cost[k]
                    * ToPrimitive::to_f64(&x[k]).unwrap().abs()
                    / 2.;
            }
            result
        });
        SmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds,
            switching_cost,
            hitting_cost: self.hitting_cost.clone(),
        }
    }
}

impl<'a, T> SmoothedLoadOptimization<T>
where
    T: Value,
{
    /// Convert instance to an instance of Smoothed Balanced-Load Optimization.
    pub fn to_sblo(&'a self) -> SmoothedBalancedLoadOptimization<'a, T> {
        let hitting_cost: Vec<CostFn<'a, T>> = self
            .hitting_cost
            .iter()
            .map(|&l| -> CostFn<'a, T> { Arc::new(move |_, _| Some(l)) })
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
    T: Value,
{
    /// Convert to an instance of Simplified Smoothed Convex Optimization.
    pub fn to_ssco(&'a self) -> SimplifiedSmoothedConvexOptimization<'a, T> {
        let f = load_balance(Arc::new(move |t, k, l| {
            self.hitting_cost[k as usize](t, NumCast::from(l[0]).unwrap())
        }));
        let loads = self
            .load
            .iter()
            .map(|l| Load::single(ToPrimitive::to_f64(l).unwrap()))
            .collect();
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: apply_loads(self.d, 1, f, loads),
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
    /// Shift a cost function to some new initial time `t_start`.
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T>;
}

impl<'a, T> ResettableCostFn<'a, T> for CostFn<'a, T> {
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T> {
        Arc::new(move |t, j| {
            let new_t = t + t_start;
            if new_t >= 1 {
                self(new_t, j)
            } else {
                Some(0.)
            }
        })
    }
}

pub trait ResettableProblem<'a, T> {
    /// Shifts problem instance to some new initial time `t_start`.
    fn reset(&'a self, t_start: i32) -> Self;
}

impl<'a, T> ResettableProblem<'a, T>
    for SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value,
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
