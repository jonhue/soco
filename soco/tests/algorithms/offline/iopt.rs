#![allow(clippy::float_cmp)]

mod make_pow_of_2 {
    use std::sync::Arc;

    use soco::algorithms::offline::iopt::make_pow_of_2;
    use soco::problem::HomProblem;

    #[test]
    fn _1() {
        let p = HomProblem {
            m: 103,
            t_end: 1_000,
            f: Arc::new(|_, _| Some(1.)),
            beta: 1.,
        };
        p.verify().unwrap();
        let transformed_p = make_pow_of_2(&p);
        transformed_p.verify().unwrap();

        assert_eq!(transformed_p.m, 128);
        assert_eq!(transformed_p.t_end, p.t_end);
        assert_eq!(transformed_p.beta, p.beta);

        for t in 1..=transformed_p.t_end {
            for j in 0..=transformed_p.m {
                assert_eq!(
                    (transformed_p.f)(t, j).unwrap(),
                    if j <= p.m {
                        1.
                    } else {
                        j as f64 * (1. + std::f64::EPSILON)
                    },
                    "f is wrongly defined for t={}, j={}",
                    t,
                    j
                );
            }
        }
    }
}

mod iopt {
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use std::sync::Arc;

    use soco::algorithms::offline::iopt::{iopt, make_pow_of_2};
    use soco::objective::Objective;
    use soco::problem::HomProblem;
    use soco::verifiers::VerifiableSchedule;

    #[test]
    fn _1() {
        let p = HomProblem {
            m: 2,
            t_end: 2,
            f: Arc::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
            beta: 1.,
        };
        p.verify().unwrap();

        let result = iopt(&p).unwrap();
        result.0.verify(p.m, p.t_end).unwrap();

        assert_eq!(result, (vec![1, 1], 1.));
        assert_eq!(result.1, p.objective_function(&result.0).unwrap());
    }

    #[test]
    fn _2() {
        let p = HomProblem {
            m: 8,
            t_end: 100,
            f: Arc::new(|t, j| {
                Some(
                    Pcg64::seed_from_u64((t * j) as u64)
                        .gen_range(0.0..1_000_000.),
                )
            }),
            beta: 1.,
        };
        p.verify().unwrap();

        let result = iopt(&p).unwrap();
        result.0.verify(p.m, p.t_end).unwrap();

        assert_eq!(result.1, p.objective_function(&result.0).unwrap());
    }

    #[test]
    fn _3() {
        let p = HomProblem {
            m: 9,
            t_end: 1_000,
            f: Arc::new(|t, j| {
                Some(
                    Pcg64::seed_from_u64((t * j) as u64)
                        .gen_range(0.0..1_000_000.),
                )
            }),
            beta: 1.,
        };
        p.verify().unwrap();

        let transformed_p = make_pow_of_2(&p);
        let result = iopt(&transformed_p).unwrap();
        result
            .0
            .verify(transformed_p.m, transformed_p.t_end)
            .unwrap();

        assert_eq!(
            result.1,
            transformed_p.objective_function(&result.0).unwrap()
        );
    }
}
