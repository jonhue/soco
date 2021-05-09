#![allow(clippy::float_cmp)]

mod make_pow_of_2 {
    use std::sync::Arc;

    use soco::algorithms::offline::iopt::make_pow_of_2;
    use soco::problem::SmoothedConvexOptimization;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 1_000,
            bounds: vec![103],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|_, _| Some(1.)),
        };
        p.verify().unwrap();
        let transformed_p = make_pow_of_2(&p).unwrap();
        transformed_p.verify().unwrap();

        assert_eq!(transformed_p.t_end, p.t_end);
        assert_eq!(transformed_p.bounds[0], 128);
        assert_eq!(transformed_p.switching_cost[0], p.switching_cost[0]);

        for t in 1..=transformed_p.t_end {
            for j in 0..=transformed_p.bounds[0] {
                assert_eq!(
                    (transformed_p.hitting_cost)(t, &vec![j]).unwrap(),
                    if j <= p.bounds[0] {
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
    use soco::problem::SmoothedConvexOptimization;
    use soco::verifiers::{VerifiableProblem, VerifiableSchedule};

    #[test]
    fn _1() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(t as f64 * (if j[0] == 0 { 1. } else { 0. }))
            }),
        };
        p.verify().unwrap();

        let result = iopt(&p).unwrap();
        result.0.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(result, (vec![vec![1], vec![1]], 1.));
        assert_eq!(result.1, p.objective_function(&result.0).unwrap());
    }

    #[test]
    fn _2() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 100,
            bounds: vec![8],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(
                    Pcg64::seed_from_u64((t * j[0]) as u64)
                        .gen_range(0.0..1_000_000.),
                )
            }),
        };
        p.verify().unwrap();

        let result = iopt(&p).unwrap();
        result.0.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(result.1, p.objective_function(&result.0).unwrap());
    }

    #[test]
    fn _3() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 1_000,
            bounds: vec![9],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(
                    Pcg64::seed_from_u64((t * j[0]) as u64)
                        .gen_range(0.0..1_000_000.),
                )
            }),
        };
        p.verify().unwrap();

        let transformed_p = make_pow_of_2(&p).unwrap();
        let result = iopt(&transformed_p).unwrap();
        result
            .0
            .verify(transformed_p.t_end, &transformed_p.bounds)
            .unwrap();

        assert_eq!(
            result.1,
            transformed_p.objective_function(&result.0).unwrap()
        );
    }
}
