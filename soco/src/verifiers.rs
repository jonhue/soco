//! Functions to check that values satisfy the imposed constraints.

use num::NumCast;

use crate::online::Online;
use crate::problem::HomProblem;
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::assert;

impl<'a, T: NumCast> HomProblem<'a, T> {
    pub fn verify(&self) -> Result<()> {
        assert_validity(self.m > 0, "m must be positive")?;
        assert_validity(self.t_end > 0, "T must be positive")?;
        assert_validity(self.beta > 0., "beta must be positive")?;

        for t in 1..=self.t_end {
            for j in 0..=self.m {
                assert_validity(
                    (self.f)(t, NumCast::from(j).unwrap()).ok_or_else(
                        || invalid("functions f must be total on their domain"),
                    )? >= 0.,
                    "functions f must be non-negative",
                )?;
            }
        }

        Ok(())
    }
}

impl<'a, T> Online<HomProblem<'a, T>>
where
    T: NumCast,
{
    pub fn verify(&self) -> Result<()> {
        assert_validity(self.w >= 0, "w must be non-negative")?;

        self.p.verify()
    }
}

pub trait VerifiableSchedule<'a, T> {
    fn verify(&self, m: i32, t_end: i32) -> Result<()>;
}

impl<'a, T: Copy + NumCast + PartialOrd> VerifiableSchedule<'a, T>
    for Schedule<T>
{
    fn verify(&self, m: i32, t_end: i32) -> Result<()> {
        assert_validity(
            self.len() == t_end as usize,
            "schedule must have a value for each time step",
        )?;

        for &x in self {
            assert_validity(
                x >= NumCast::from(0).unwrap(),
                "values in schedule must be non-negative",
            )?;
            assert_validity(
                x <= NumCast::from(m).unwrap(),
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
