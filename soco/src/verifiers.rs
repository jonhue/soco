//! Functions to check that values satisfy the imposed constraints.

use num::NumCast;

use crate::cost::CostFn;
use crate::online::Online;
use crate::problem::{SmoothedConvexOptimization, SmoothedLoadOptimization};
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::assert;

pub trait VerifiableCostFn<'a, T> {
    fn verify(&self, t: i32, x: &T) -> Result<()>;
}

impl<'a, T> VerifiableCostFn<'a, T> for CostFn<'a, T>
where
    T: std::fmt::Debug,
{
    fn verify(&self, t: i32, x: &T) -> Result<()> {
        assert_validity(
            self(t, x).ok_or_else(|| {
                invalid(format!("cost function must be total on its domain, but returns None for ({}, {:?})", t, x))
            })? >= 0.,
            format!("cost function must be non-negative, but is for ({}, {:?})", t, x),
        )
    }
}

pub trait VerifiableProblem {
    fn verify(&self) -> Result<()>;
}

impl<'a, T> VerifiableProblem for SmoothedConvexOptimization<'a, T>
where
    T: Clone + NumCast + PartialOrd + std::fmt::Debug,
{
    fn verify(&self) -> Result<()> {
        assert_validity(
            self.d > 0,
            format!("number of dimensions must be positive, is {}", self.d),
        )?;
        assert_validity(
            self.t_end > 0,
            format!("time horizon must be positive, is {}", self.t_end),
        )?;
        assert_validity(
            self.bounds.len() == self.d as usize,
            format!("length of vector of upper bounds must equal dimension, {} != {}", self.bounds.len(), self.d),
        )?;
        assert_validity(
            self.switching_cost.len() == self.d as usize,
            format!("length of vector of switching cost factors must equal dimension, {} != {}", self.switching_cost.len(), self.d),
        )?;

        for k in 0..self.d as usize {
            assert_validity(
                self.bounds[k] > NumCast::from(0).unwrap(),
                format!("upper bound of dimension {} must be positive", k + 1),
            )?;
            assert_validity(
                self.switching_cost[k] > 0.,
                format!(
                    "switching cost of dimension {} must be positive",
                    k + 1
                ),
            )?;

            for t in 1..=self.t_end {
                self.hitting_cost.verify(
                    t,
                    &vec![NumCast::from(0).unwrap(); self.d as usize],
                )?;
                self.hitting_cost.verify(t, &self.bounds)?;
            }
        }

        Ok(())
    }
}

impl<T> VerifiableProblem for SmoothedLoadOptimization<T>
where
    T: Clone + NumCast + PartialOrd + std::fmt::Debug,
{
    fn verify(&self) -> Result<()> {
        assert_validity(
            self.d > 0,
            format!("number of dimensions must be positive, is {}", self.d),
        )?;
        assert_validity(
            self.t_end > 0,
            format!("time horizon must be positive, is {}", self.t_end),
        )?;
        assert_validity(
            self.bounds.len() == self.d as usize,
            format!("length of vector of upper bounds must equal dimension, {} != {}", self.bounds.len(), self.d),
        )?;
        assert_validity(
            self.switching_cost.len() == self.d as usize,
            format!("length of vector of switching cost factors must equal dimension, {} != {}", self.switching_cost.len(), self.d),
        )?;

        for k in 0..self.d as usize {
            assert_validity(
                self.bounds[k] > NumCast::from(0).unwrap(),
                format!("upper bound of dimension {} must be positive", k + 1),
            )?;
            assert_validity(
                self.switching_cost[k] > 0.,
                format!(
                    "switching cost of dimension {} must be positive",
                    k + 1
                ),
            )?;
            assert_validity(
                self.hitting_cost[k] >= 0.,
                format!(
                    "hitting cost of dimension {} must be non-negative",
                    k + 1
                ),
            )?;

            for l in 0..self.d as usize {
                if k != l {
                    assert_validity(
                        !(self.hitting_cost[k] >= self.hitting_cost[l] &&
                            self.switching_cost[k] >= self.switching_cost[l]),
                        format!(
                            "dimension {} is inefficient compared to dimension {}",
                            k + 1, l + 1
                        ),
                    )?;
                }
            }
        }

        for k in 1..self.d as usize {
            assert_validity(
                self.hitting_cost[k] > self.hitting_cost[k - 1],
                format!(
                    "hitting costs must be ascending, are not between dimension {} and dimension {}", k,
                    k + 1
                ),
            )?;
            assert_validity(
                self.switching_cost[k] < self.switching_cost[k - 1],
                format!(
                    "hitting costs must be descending, are not between dimension {} and dimension {}", k,
                    k + 1
                ),
            )?;
        }

        Ok(())
    }
}

impl<'a, T> Online<T>
where
    T: VerifiableProblem,
{
    pub fn verify(&self) -> Result<()> {
        assert_validity(
            self.w >= 0,
            format!("w must be non-negative, is {}", self.w),
        )?;

        self.p.verify()
    }
}

pub trait VerifiableSchedule<'a, T> {
    fn verify(&self, t_end: i32, bounds: &Vec<T>) -> Result<()>;
}

impl<'a, T> VerifiableSchedule<'a, T> for Schedule<T>
where
    T: Copy + NumCast + PartialOrd,
{
    fn verify(&self, t_end: i32, bounds: &Vec<T>) -> Result<()> {
        assert_validity(
            self.len() == t_end as usize,
            format!("schedule must have a value for each time step, `t_end` is {} and schedule contains {} steps", t_end, self.len()),
        )?;

        for (t, x) in self.iter().enumerate() {
            for (k, &j) in x.iter().enumerate() {
                assert_validity(
                    j >= NumCast::from(0).unwrap(),
                    format!("value at time {} for dimension {} must be non-negative", t + 1, k + 1),
                )?;
                assert_validity(
                    j <= bounds[k],
                    format!("value at time {} for dimension {} must not exceed its upper bound", t + 1, k + 1),
                )?;
            }
        }

        Ok(())
    }
}

fn assert_validity(pred: bool, message: String) -> Result<()> {
    assert(pred, invalid(message))
}

fn invalid(message: String) -> Error {
    Error::Invalid(message)
}
