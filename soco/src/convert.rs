//! Functions to convert between problem instances.

use std::sync::Arc;

use crate::cost::CostFn;
use crate::online::Online;
use crate::problem::{ContinuousHomProblem, DiscreteHomProblem, HomProblem};
use crate::schedule::{ContinuousSchedule, DiscreteSchedule};

pub trait DiscretizableCostFn<'a> {
    fn to_i(&'a self) -> CostFn<'a, i32>;
}

impl<'a> DiscretizableCostFn<'a> for CostFn<'a, f64> {
    fn to_i(&'a self) -> CostFn<'a, i32> {
        Arc::new(move |t, j: i32| self(t, j as f64))
    }
}

pub trait ExtendableCostFn<'a> {
    fn to_f(&'a self) -> CostFn<'a, f64>;
}

impl<'a> ExtendableCostFn<'a> for CostFn<'a, i32> {
    fn to_f(&'a self) -> CostFn<'a, f64> {
        Arc::new(move |t, j: f64| {
            if j.fract() == 0. {
                self(t, j as i32)
            } else {
                let l = self(t, j.floor() as i32);
                let u = self(t, j.ceil() as i32);
                if l.is_none() || u.is_none() {
                    return None;
                }

                Some((j.ceil() - j) * l.unwrap() + (j - j.floor()) * u.unwrap())
            }
        })
    }
}

impl<'a> ContinuousHomProblem<'a> {
    fn to_i(&'a self) -> DiscreteHomProblem<'a> {
        HomProblem {
            m: self.m,
            t_end: self.t_end,
            beta: self.beta,
            f: self.f.to_i(),
        }
    }
}

impl<'a> DiscreteHomProblem<'a> {
    fn to_f(&'a self) -> ContinuousHomProblem<'a> {
        HomProblem {
            m: self.m,
            t_end: self.t_end,
            beta: self.beta,
            f: self.f.to_f(),
        }
    }
}

impl<'a> Online<ContinuousHomProblem<'a>> {
    pub fn to_i(&'a self) -> Online<DiscreteHomProblem<'a>> {
        Online {
            w: self.w,
            p: self.p.to_i(),
        }
    }
}

impl<'a> Online<DiscreteHomProblem<'a>> {
    pub fn to_f(&'a self) -> Online<ContinuousHomProblem<'a>> {
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
        self.iter().map(|&x| x.ceil() as i32).collect()
    }
}

pub trait ExtendableSchedule {
    fn to_f(&self) -> ContinuousSchedule;
}

impl ExtendableSchedule for DiscreteSchedule {
    fn to_f(&self) -> ContinuousSchedule {
        self.iter().map(|&x| x as f64).collect()
    }
}
