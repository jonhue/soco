//! Functions to convert between problem instances.

use std::sync::Arc;

use crate::problem::{
    ContinuousHomProblem, DiscreteHomProblem, HomProblem, Online,
};

impl<'a> ContinuousHomProblem<'a> {
    /// Converts a continuous problem instance to a discrete one.
    pub fn to_i(&'a self) -> DiscreteHomProblem<'a> {
        HomProblem {
            m: self.m,
            t_end: self.t_end,
            beta: self.beta,
            f: Arc::new(move |t, j: i32| (self.f)(t, j as f64)),
        }
    }
}

impl<'a> DiscreteHomProblem<'a> {
    /// Converts a discrete problem instance to a continuous one.
    pub fn to_f(&'a self) -> ContinuousHomProblem<'a> {
        HomProblem {
            m: self.m,
            t_end: self.t_end,
            beta: self.beta,
            f: Arc::new(move |t, j: f64| {
                if j.fract() == 0. {
                    (self.f)(t, j as i32)
                } else {
                    let l = (self.f)(t, j.floor() as i32);
                    let u = (self.f)(t, j.ceil() as i32);
                    if l.is_none() || u.is_none() {
                        return None;
                    }

                    Some(
                        (j.ceil() - j) * l.unwrap()
                            + (j - j.floor()) * u.unwrap(),
                    )
                }
            }),
        }
    }
}

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Converts a continuous online problem instance to a discrete one.
    pub fn to_i(&'a self) -> Online<DiscreteHomProblem<'a>> {
        Online {
            w: self.w,
            p: self.p.to_i(),
        }
    }
}

impl<'a> Online<DiscreteHomProblem<'a>> {
    /// Converts a discrete online problem instance to a continuous one.
    pub fn to_f(&'a self) -> Online<ContinuousHomProblem<'a>> {
        Online {
            w: self.w,
            p: self.p.to_f(),
        }
    }
}
