//! Functions to check that values satisfy the imposed constraints.

use crate::online::Online;
use crate::problem::{ContinuousHomProblem, DiscreteHomProblem};
use crate::result::{Error, Result};
use crate::schedule::{ContinuousSchedule, DiscreteSchedule};
use crate::utils::assert;

pub trait VerifiableProblem {
    fn verify(&self) -> Result<()>;
}

impl<'a> VerifiableProblem for DiscreteHomProblem<'a> {
    fn verify(&self) -> Result<()> {
        assert_validity(self.m > 0, "m must be positive")?;
        assert_validity(self.t_end > 0, "T must be positive")?;
        assert_validity(self.beta > 0., "beta must be positive")?;

        for t in 1..=self.t_end {
            for j in 0..=self.m {
                assert_validity(
                    (self.f)(t, j).ok_or_else(|| invalid(
                        "functions f must be total on their domain",
                    ))? >= 0.,
                    "functions f must be non-negative",
                )?;
            }
        }

        Ok(())
    }
}

impl<'a> VerifiableProblem for ContinuousHomProblem<'a> {
    fn verify(&self) -> Result<()> {
        assert_validity(self.m > 0, "m must be positive")?;
        assert_validity(self.t_end > 0, "T must be positive")?;
        assert_validity(self.beta > 0., "beta must be positive")?;

        for t in 1..=self.t_end {
            for j in 0..=self.m {
                assert_validity(
                    (self.f)(t, j as f64).ok_or_else(|| invalid(
                        "functions f must be total on their domain",
                    ))? >= 0.,
                    "functions f must be non-negative",
                )?;
            }
        }

        Ok(())
    }
}

impl<T> Online<T>
where
    T: VerifiableProblem,
{
    pub fn verify(&self) -> Result<()> {
        assert_validity(self.w >= 0, "w must be non-negative")?;

        self.p.verify()
    }
}

pub trait VerifiableSchedule<'a, T> {
    fn verify(&self, m: i32, t_end: i32) -> Result<()>;
}

impl<'a> VerifiableSchedule<'a, i32> for DiscreteSchedule {
    fn verify(&self, m: i32, t_end: i32) -> Result<()> {
        assert_validity(
            self.len() == t_end as usize,
            "schedule must have a value for each time step",
        )?;

        for &x in self {
            assert_validity(x >= 0, "values in schedule must be non-negative")?;
            assert_validity(
                x <= m,
                "values in schedule must not exceed the number of servers",
            )?;
        }

        Ok(())
    }
}

impl<'a> VerifiableSchedule<'a, f64> for ContinuousSchedule {
    fn verify(&self, m: i32, t_end: i32) -> Result<()> {
        assert_validity(
            self.len() == t_end as usize,
            "schedule must have a value for each time step",
        )?;

        for &x in self {
            assert_validity(
                x >= 0.,
                "values in schedule must be non-negative",
            )?;
            assert_validity(
                x <= m as f64,
                "values in schedule must not exceed the number of servers",
            )?;
        }

        Ok(())
    }
}

fn assert_validity(pred: bool, message: &str) -> Result<()> {
    assert(pred, invalid(message))
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.to_string())
}
