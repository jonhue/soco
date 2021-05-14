//! Functions to convert between problem instances.

use num::{Num, NumCast, ToPrimitive};
use std::sync::Arc;

use crate::config::Config;
use crate::cost::CostFn;
use crate::cost::{lazy, LoadCostFn};
use crate::online::Online;
use crate::problem::{
    FractionalSmoothedConvexOptimization, IntegralSmoothedConvexOptimization,
    SmoothedBalancedLoadOptimization, SmoothedConvexOptimization,
    SmoothedLoadOptimization,
};
use crate::schedule::{FractionalSchedule, IntegralSchedule};
use crate::vec_wrapper::VecWrapper;

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
    fn to_i(&'a self) -> CostFn<'a, Vec<i32>>;
}

impl<'a> DiscretizableCostFn<'a> for CostFn<'a, Vec<f64>> {
    fn to_i(&'a self) -> CostFn<'a, Vec<i32>> {
        Arc::new(move |t, x| self(t, x.to_f()))
    }
}

pub trait RelaxableCostFn<'a> {
    /// Relax an integral cost function to the fractional setting.
    fn to_f(&'a self) -> CostFn<'a, Vec<f64>>;
}

impl<'a> RelaxableCostFn<'a> for CostFn<'a, Vec<i32>> {
    fn to_f(&'a self) -> CostFn<'a, Vec<f64>> {
        Arc::new(move |t, x| {
            assert!(x.len() == 1, "cannot relax multidimensional problems");

            let j = x[0];
            if j.fract() == 0. {
                self(t, vec![j as i32])
            } else {
                let l = self(t, vec![j.floor() as i32]);
                let u = self(t, vec![j.ceil() as i32]);
                if l.is_none() || u.is_none() {
                    return None;
                }

                Some((j.ceil() - j) * l.unwrap() + (j - j.floor()) * u.unwrap())
            }
        })
    }
}

pub trait DiscretizableProblem<'a> {
    type Output;

    /// Discretize a fractional problem instance.
    fn to_i(&'a self) -> Self::Output;
}

impl<'a> DiscretizableProblem<'a> for FractionalSmoothedConvexOptimization<'a> {
    type Output = IntegralSmoothedConvexOptimization<'a>;

    fn to_i(&'a self) -> IntegralSmoothedConvexOptimization<'a> {
        SmoothedConvexOptimization {
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

impl<'a> RelaxableProblem<'a> for IntegralSmoothedConvexOptimization<'a> {
    type Output = FractionalSmoothedConvexOptimization<'a>;

    fn to_f(&'a self) -> FractionalSmoothedConvexOptimization<'a> {
        SmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.to_f(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.to_f(),
        }
    }
}

impl<'a, T> SmoothedLoadOptimization<T>
where
    T: Clone + Copy + NumCast,
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
    T: Clone + Copy + Num + NumCast,
{
    /// Convert instance to an instance of Smoothed Convex Optimization.
    pub fn to_sco(&'a self) -> SmoothedConvexOptimization<'a, T> {
        let f: LoadCostFn<'a, T> = Arc::new(move |t, k, l| {
            Arc::new(move |j| match self.hitting_cost[k as usize](t, l / j) {
                None => None,
                Some(hitting_cost) => {
                    Some(ToPrimitive::to_f64(&j).unwrap() * hitting_cost)
                }
            })
        });
        SmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: lazy(self.d, f, &self.load),
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
    fn ceil(&self) -> Config<i32>;
    /// Floor all elements of a config.
    fn floor(&self) -> Config<i32>;
}

impl DiscretizableConfig for Config<f64> {
    fn ceil(&self) -> Config<i32> {
        Config::new(self.to_vec().ceil())
    }

    fn floor(&self) -> Config<i32> {
        Config::new(self.to_vec().floor())
    }
}

pub trait RelaxableConfig {
    /// Convert an integral config to a fractional config.
    fn to_f(&self) -> Config<f64>;
}

impl RelaxableConfig for Config<i32> {
    fn to_f(&self) -> Config<f64> {
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
        Arc::new(move |t, j| self(t + t_start, j))
    }
}

impl<'a, T> SmoothedConvexOptimization<'a, T>
where
    T: Clone,
{
    /// Shifts problem instance to some new initial time `t_start`.
    pub fn reset(&'a self, t_start: i32) -> SmoothedConvexOptimization<'a, T> {
        SmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end - t_start,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.reset(t_start),
        }
    }
}
