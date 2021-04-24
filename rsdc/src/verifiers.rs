//! Functions to check that values satisfy the imposed constraints.

use crate::problem::{ContinuousHomProblem, ContinuousSchedule, DiscreteHomProblem, DiscreteSchedule, HomProblem, Online};

pub trait VerifiableProblem {
    fn verify(&self);
}

impl<'a> VerifiableProblem for DiscreteHomProblem<'a> {
    fn verify(&self) {
        assert!(self.m > 0, "m must be positive");
        assert!(self.t_end > 0, "T must be positive");
        assert!(self.beta > 0., "beta must be positive");

        for t in 1..=self.t_end {
            for j in 0..=self.m {
                assert!(
                    (self.f)(t, j)
                        .expect("functions f must be total on their domain")
                        >= 0.,
                    "functions f must be non-negative"
                );
            }
        }
    }
}

impl<'a> VerifiableProblem for ContinuousHomProblem<'a> {
    fn verify(&self) {
        assert!(self.m > 0, "m must be positive");
        assert!(self.t_end > 0, "T must be positive");
        assert!(self.beta > 0., "beta must be positive");

        for t in 1..=self.t_end {
            for j in 0..=self.m {
                assert!(
                    (self.f)(t, j as f64)
                        .expect("functions f must be total on their domain")
                        >= 0.,
                    "functions f must be non-negative"
                );
            }
        }
    }
}

impl<T> Online<T>
where
    T: VerifiableProblem,
{
    pub fn verify(&self) {
        assert!(self.w >= 0, "w must be non-negative");

        self.p.verify();
    }
}

pub trait VerifiableSchedule<'a, T> {
    fn verify(&self, p: &HomProblem<'a, T>);
}

impl<'a> VerifiableSchedule<'a, i32> for DiscreteSchedule {
    fn verify(&self, p: &HomProblem<'a, i32>) {
        assert_eq!(
            self.len(),
            p.t_end as usize,
            "schedule must have a value for each time step"
        );

        for &x in self {
            assert!(x >= 0, "values in schedule must be non-negative");
            assert!(
                x <= p.m,
                "values in schedule must not exceed the number of servers"
            );
        }
    }
}

impl<'a> VerifiableSchedule<'a, f64> for ContinuousSchedule {
    fn verify(&self, p: &HomProblem<'a, f64>) {
        assert_eq!(
            self.len(),
            p.t_end as usize,
            "schedule must have a value for each time step"
        );

        for &x in self {
            assert!(x >= 0., "values in schedule must be non-negative");
            assert!(
                x <= p.m as f64,
                "values in schedule must not exceed the number of servers"
            );
        }
    }
}
