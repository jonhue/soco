//! Functions to check that values satisfy the imposed constraints.

use crate::config::Config;
use crate::problem::Online;
use crate::problem::{
    SimplifiedSmoothedConvexOptimization, SmoothedBalancedLoadOptimization,
    SmoothedConvexOptimization, SmoothedLoadOptimization,
};
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use num::NumCast;

pub trait VerifiableProblem {
    fn verify(&self) -> Result<()>;
}

impl<'a, T, C, D> VerifiableProblem for SmoothedConvexOptimization<'a, T, C, D>
where
    T: Value<'a>,
{
    fn verify(&self) -> Result<()> {
        assert_validity(
            self.d > 0,
            format!("number of dimensions must be positive, is {}", self.d),
        )?;
        assert_validity(
            self.t_end >= 0,
            format!("time horizon must be non-negative, is {}", self.t_end),
        )?;
        assert_validity(
            self.bounds.len() == self.d as usize,
            format!("length of vector of upper bounds must equal dimension, {} != {}", self.bounds.len(), self.d),
        )?;

        Ok(())
    }
}

impl<'a, T, C, D> VerifiableProblem
    for SimplifiedSmoothedConvexOptimization<'a, T, C, D>
where
    T: Value<'a>,
{
    fn verify(&self) -> Result<()> {
        assert_validity(
            self.d > 0,
            format!("number of dimensions must be positive, is {}", self.d),
        )?;
        assert_validity(
            self.t_end >= 0,
            format!("time horizon must be non-negative, is {}", self.t_end),
        )?;
        assert_validity(
            self.bounds.len() == self.d as usize,
            format!("length of vector of upper bounds must equal dimension, {} != {}", self.bounds.len(), self.d),
        )?;
        assert_validity(
            self.switching_cost.len() == self.d as usize,
            format!("length of vector of switching costs must equal dimension, {} != {}", self.switching_cost.len(), self.d),
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
        }

        Ok(())
    }
}

impl<'a, T> VerifiableProblem for SmoothedLoadOptimization<T>
where
    T: Value<'a>,
{
    fn verify(&self) -> Result<()> {
        assert_validity(
            self.d > 0,
            format!("number of dimensions must be positive, is {}", self.d),
        )?;
        assert_validity(
            self.t_end >= 0,
            format!("time horizon must be non-negative, is {}", self.t_end),
        )?;
        assert_validity(
            self.bounds.len() == self.d as usize,
            format!("length of vector of upper bounds must equal dimension, {} != {}", self.bounds.len(), self.d),
        )?;
        assert_validity(
            self.switching_cost.len() == self.d as usize,
            format!("length of vector of switching costs must equal dimension, {} != {}", self.switching_cost.len(), self.d),
        )?;
        assert_validity(
            self.load.len() >= self.t_end as usize,
            format!(
                "length of vector of loads must be sufficient for time horizon, {} < {}",
                self.load.len(),
                self.t_end
            ),
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
                self.hitting_cost[k] < self.hitting_cost[k - 1],
                format!(
                    "hitting costs must be descending, are not between dimension {} and dimension {}", k,
                    k + 1
                ),
            )?;
            assert_validity(
                self.switching_cost[k] > self.switching_cost[k - 1],
                format!(
                    "switching costs must be ascending, are not between dimension {} and dimension {}", k,
                    k + 1
                ),
            )?;
        }

        for t in 0..self.t_end as usize {
            assert_validity(
                self.load[t] >= NumCast::from(0).unwrap(),
                format!(
                    "load must be non-negative, is {} at time {}",
                    self.load[t],
                    t + 1
                ),
            )?;
        }

        Ok(())
    }
}

impl<'a, T> VerifiableProblem for SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value<'a>,
{
    fn verify(&self) -> Result<()> {
        assert_validity(
            self.d > 0,
            format!("number of dimensions must be positive, is {}", self.d),
        )?;
        assert_validity(
            self.t_end >= 0,
            format!("time horizon must be non-negative, is {}", self.t_end),
        )?;
        assert_validity(
            self.bounds.len() == self.d as usize,
            format!("length of vector of upper bounds must equal dimension, {} != {}", self.bounds.len(), self.d),
        )?;
        assert_validity(
            self.switching_cost.len() == self.d as usize,
            format!("length of vector of switching costs must equal dimension, {} != {}", self.switching_cost.len(), self.d),
        )?;
        assert_validity(
            self.load.len() >= self.t_end as usize,
            format!(
                "length of vector of loads must be sufficient for time horizon, {} != {}",
                self.load.len(),
                self.t_end
            ),
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
        }

        for t in 0..self.t_end as usize {
            assert_validity(
                self.load[t] >= NumCast::from(0).unwrap(),
                format!(
                    "load must be non-negative, is {} at time {}",
                    self.load[t],
                    t + 1
                ),
            )?;
        }

        Ok(())
    }
}

impl<'a, P> Online<P>
where
    P: VerifiableProblem,
{
    pub fn verify(&self) -> Result<()> {
        assert_validity(
            self.w >= 0,
            format!("w must be non-negative, is {}", self.w),
        )?;

        self.p.verify()
    }
}

impl<'a, T> Config<T>
where
    T: Value<'a>,
{
    pub fn verify(&self, t: i32, bounds: &Vec<T>) -> Result<()> {
        for (k, &j) in self.iter().enumerate() {
            assert_validity(
                j >= NumCast::from(0).unwrap(),
                format!(
                    "value at time {} for dimension {} must be non-negative",
                    t + 1,
                    k + 1
                ),
            )?;
            assert_validity(
                j <= bounds[k],
                format!("value at time {} for dimension {} must not exceed its upper bound", t + 1, k + 1),
            )?;
        }

        Ok(())
    }
}

impl<'a, T> Schedule<T>
where
    T: Value<'a>,
{
    pub fn verify(&self, t_end: i32, bounds: &Vec<T>) -> Result<()> {
        assert_validity(
            self.t_end() == t_end,
            format!("schedule must have a value for each time step, `t_end` is {} and schedule contains {} steps", t_end, self.t_end()),
        )?;

        for (t, x) in self.iter().enumerate() {
            x.verify(t as i32, bounds)?;
        }

        Ok(())
    }
}

fn assert_validity(pred: bool, message: String) -> Result<()> {
    assert(pred, invalid(message))
}

fn invalid(message: String) -> Failure {
    Failure::Invalid(message)
}
