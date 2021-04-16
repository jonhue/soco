use crate::lib::types::{DiscreteHomProblem, DiscreteSchedule};

pub fn verify_discrete_problem(p: &DiscreteHomProblem) {
    assert!(p.m >= 0, "m must be non-negative");
    assert!(p.t_end > 0, "T must be positive");
    assert!(p.beta > 0., "beta must be positive");

    for t in 1..p.t_end {
        for j in 0..p.m {
            assert_ne!(
                (p.f)(t, j),
                None,
                "functions f must be total on their domain"
            );
        }
    }
}

pub fn verify_discrete_schedule(p: &DiscreteHomProblem, xs: &DiscreteSchedule) {
    assert_eq!(
        xs.len(),
        p.t_end as usize,
        "schedule must have a value for each time step"
    );

    for x in xs {
        assert!(x >= &0, "values in schedule must be non-negative");
        assert!(
            x <= &p.m,
            "values in schedule must not exceed the number of servers"
        );
    }
}
