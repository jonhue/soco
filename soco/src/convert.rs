//! Functions to convert between problem instances.

use std::sync::Arc;

use crate::cost::CostFn;
use crate::online::Online;
use crate::problem::{ContinuousProblem, DiscreteProblem, Problem};
use crate::schedule::{ContinuousSchedule, DiscreteSchedule};

pub trait DiscretizableVector {
    fn ceil(&self) -> Vec<i32>;
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
    fn to_f(&self) -> Vec<f64>;
}

impl RelaxableVector for Vec<i32> {
    fn to_f(&self) -> Vec<f64> {
        self.iter().map(|&x| x as f64).collect()
    }
}

pub trait DiscretizableCostFn<'a> {
    fn to_i(&'a self) -> CostFn<'a, Vec<i32>>;
}

impl<'a> DiscretizableCostFn<'a> for CostFn<'a, Vec<f64>> {
    fn to_i(&'a self) -> CostFn<'a, Vec<i32>> {
        Arc::new(move |t, x| self(t, &x.to_f()))
    }
}

pub trait RelaxableCostFn<'a> {
    fn to_f(&'a self) -> CostFn<'a, Vec<f64>>;
}

impl<'a> RelaxableCostFn<'a> for CostFn<'a, Vec<i32>> {
    fn to_f(&'a self) -> CostFn<'a, Vec<f64>> {
        Arc::new(move |t, x| {
            assert!(x.len() == 1, "cannot relax multidimensional problems");

            let j = x[0];
            if j.fract() == 0. {
                self(t, &vec![j as i32])
            } else {
                let l = self(t, &vec![j.floor() as i32]);
                let u = self(t, &vec![j.ceil() as i32]);
                if l.is_none() || u.is_none() {
                    return None;
                }

                Some((j.ceil() - j) * l.unwrap() + (j - j.floor()) * u.unwrap())
            }
        })
    }
}

impl<'a> ContinuousProblem<'a> {
    pub fn to_i(&'a self) -> DiscreteProblem<'a> {
        Problem {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.floor(),
            switching_costs: self.switching_costs.clone(),
            f: self.f.to_i(),
        }
    }
}

impl<'a> DiscreteProblem<'a> {
    pub fn to_f(&'a self) -> ContinuousProblem<'a> {
        Problem {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.to_f(),
            switching_costs: self.switching_costs.clone(),
            f: self.f.to_f(),
        }
    }
}

impl<'a> Online<ContinuousProblem<'a>> {
    pub fn to_i(&'a self) -> Online<DiscreteProblem<'a>> {
        Online {
            w: self.w,
            p: self.p.to_i(),
        }
    }
}

impl<'a> Online<DiscreteProblem<'a>> {
    pub fn to_f(&'a self) -> Online<ContinuousProblem<'a>> {
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

impl<'a, T> Problem<'a, T>
where
    T: Clone,
{
    pub fn reset(&'a self, t_start: i32) -> Problem<'a, T> {
        Problem {
            d: self.d,
            t_end: self.t_end - t_start,
            bounds: self.bounds.clone(),
            switching_costs: self.switching_costs.clone(),
            f: self.f.reset(t_start),
        }
    }
}
