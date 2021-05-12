//! Functions to convert between problem instances.

use crate::cost::{lazy, LazyCostFn};
use num::{NumCast, ToPrimitive};
use std::sync::Arc;

use crate::cost::CostFn;
use crate::online::Online;
use crate::problem::{
    ContinuousSmoothedConvexOptimization, DiscreteSmoothedConvexOptimization,
    SmoothedConvexOptimization, SmoothedLoadOptimization,
};
use crate::schedule::{ContinuousSchedule, DiscreteSchedule};

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
    /// Convert a discrete vector to a continuous vector.
    fn to_f(&self) -> Vec<f64>;
}

impl RelaxableVector for Vec<i32> {
    fn to_f(&self) -> Vec<f64> {
        self.iter().map(|&x| x as f64).collect()
    }
}

pub trait DiscretizableCostFn<'a> {
    /// Discretize a continuous cost function.
    fn to_i(&'a self) -> CostFn<'a, Vec<i32>>;
}

impl<'a> DiscretizableCostFn<'a> for CostFn<'a, Vec<f64>> {
    fn to_i(&'a self) -> CostFn<'a, Vec<i32>> {
        Arc::new(move |t, x| self(t, x.to_f()))
    }
}

pub trait RelaxableCostFn<'a> {
    /// Relax a discrete cost function to the continuous setting.
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

impl<'a> ContinuousSmoothedConvexOptimization<'a> {
    /// Discretize a problem instance.
    pub fn to_i(&'a self) -> DiscreteSmoothedConvexOptimization<'a> {
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
    /// Relax a discrete problem instance to the continuous setting.
    fn to_f(&'a self) -> CostFn<'a, Vec<f64>>;
}

impl<'a> DiscreteSmoothedConvexOptimization<'a> {
    /// Relax a problem instance.
    pub fn to_f(&'a self) -> ContinuousSmoothedConvexOptimization<'a> {
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
    /// Convert instance to an instance of Smoothed Convex Optimization.
    pub fn to_sco(&'a self) -> SmoothedConvexOptimization<'a, T> {
        let f: LazyCostFn<'a, T> = Arc::new(|l| {
            Arc::new(move |x| {
                let prim_l = ToPrimitive::to_f64(&l).unwrap();
                let prim_x = ToPrimitive::to_f64(&x).unwrap();
                if prim_x >= prim_l {
                    Some(prim_l * prim_x)
                } else {
                    Some(f64::INFINITY)
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

impl<'a> Online<ContinuousSmoothedConvexOptimization<'a>> {
    pub fn to_i(&'a self) -> Online<DiscreteSmoothedConvexOptimization<'a>> {
        Online {
            w: self.w,
            p: self.p.to_i(),
        }
    }
}

impl<'a> Online<DiscreteSmoothedConvexOptimization<'a>> {
    pub fn to_f(&'a self) -> Online<ContinuousSmoothedConvexOptimization<'a>> {
        Online {
            w: self.w,
            p: self.p.to_f(),
        }
    }
}

pub trait DiscretizableSchedule {
    fn to_i(&self) -> DiscreteSchedule;
}

impl DiscretizableSchedule for ContinuousSchedule {
    fn to_i(&self) -> DiscreteSchedule {
        self.iter().map(|x| x.ceil()).collect()
    }
}

pub trait RelaxableSchedule {
    fn to_f(&self) -> ContinuousSchedule;
}

impl RelaxableSchedule for DiscreteSchedule {
    fn to_f(&self) -> ContinuousSchedule {
        self.iter().map(|x| x.to_f()).collect()
    }
}

pub trait ResettableCostFn<'a, T> {
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
