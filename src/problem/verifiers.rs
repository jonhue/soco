//! Functions to check that values satisfy the imposed constraints.

use crate::problem::types::{DiscreteHomProblem, DiscreteSchedule, Online};

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

impl<T> Online<T>
where
    T: VerifiableProblem,
{
    pub fn verify(&self) {
        assert!(self.w >= 0, "w must be non-negative");

        self.p.verify();
    }
}

pub trait VerifiableSchedule {
    fn verify(&self, p: &DiscreteHomProblem);
}

impl VerifiableSchedule for DiscreteSchedule {
    fn verify(&self, p: &DiscreteHomProblem) {
        assert_eq!(
            self.len(),
            p.t_end as usize,
            "schedule must have a value for each time step"
        );

        for x in self {
            assert!(x >= &0, "values in schedule must be non-negative");
            assert!(
                x <= &p.m,
                "values in schedule must not exceed the number of servers"
            );
        }
    }
}
