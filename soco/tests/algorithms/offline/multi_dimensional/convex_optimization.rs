#[cfg(test)]
mod co {
    use crate::init;
    use crate::{factories::inv_e, utils::upper_bounds};
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::algorithms::offline::{
        multi_dimensional::convex_optimization::co,
        OfflineAlgorithmWithDefaultOptions,
    };
    use soco::algorithms::offline::{OfflineOptions, OfflineResult};
    use soco::config::Config;
    use soco::convert::{CastableSchedule, DiscretizableSchedule};
    use soco::norm::{euclidean, manhattan_scaled, norm_squared};
    use soco::objective::Objective;
    use soco::problem::SmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

        let p = SmoothedConvexOptimization {
            d: 2,
            t_end: 2,
            bounds: vec![(0., 2.), (0., 1.)],
            switching_cost: manhattan_scaled(vec![1.5, 1.]),
            hitting_cost: inv_e(),
        };
        p.verify().unwrap();

        let result = co
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap()
            .xs();
        result.verify(p.t_end, &upper_bounds(&p.bounds)).unwrap();

        let int_result = result.to_i();
        let cast_int_result: Schedule<f64> = int_result.to();
        assert_eq!(
            int_result,
            Schedule::new(vec![
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1])
            ])
        );
        assert_abs_diff_eq!(
            p.objective_function(&cast_int_result).unwrap().raw(),
            3.5096,
            epsilon = 1e-4
        );
    }

    #[test]
    fn _2() {
        init();

        let p = SmoothedConvexOptimization {
            d: 2,
            t_end: 100,
            bounds: vec![(0., 8.), (0., 8.)],
            switching_cost: euclidean(),
            hitting_cost: inv_e(),
        };
        p.verify().unwrap();

        let result = co
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap()
            .xs();
        result.verify(p.t_end, &upper_bounds(&p.bounds)).unwrap();
    }

    #[test]
    fn _3() {
        init();

        let euclidean_ = euclidean();

        let d = 4;
        let t_end = 10;
        let p = SmoothedConvexOptimization {
            d,
            t_end,
            bounds: (0..d)
                .map(|_| {
                    (
                        0.,
                        Pcg64::seed_from_u64((d * t_end) as u64).gen_range(1..5)
                            as f64,
                    )
                })
                .collect(),
            switching_cost: norm_squared(&euclidean_),
            hitting_cost: inv_e(),
        };
        p.verify().unwrap();

        let result = co
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap()
            .xs();
        result.verify(p.t_end, &upper_bounds(&p.bounds)).unwrap();
    }

    #[test]
    fn _4() {
        init();

        let euclidean_ = euclidean();
        let epsilon = 0.00001;

        let d = 4;
        let t_end = 5;
        let p = SmoothedConvexOptimization {
            d,
            t_end,
            bounds: (0..d)
                .map(|_| {
                    (
                        0.,
                        Pcg64::seed_from_u64((d * t_end) as u64).gen_range(1..5)
                            as f64,
                    )
                })
                .collect(),
            switching_cost: norm_squared(&euclidean_),
            hitting_cost: inv_e(),
        };
        p.verify().unwrap();

        let l = 1.;
        let result = co
            .solve_with_default_options(
                p.clone(),
                OfflineOptions::l_constrained(l),
            )
            .unwrap()
            .xs();
        result.verify(p.t_end, &upper_bounds(&p.bounds)).unwrap();
        assert!(p.total_movement(&result, false).unwrap().raw() <= l + epsilon);
    }
}
