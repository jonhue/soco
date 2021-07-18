mod co {
    use crate::factories::{inv_e};
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::algorithms::offline::{
        multi_dimensional::convex_optimization::co,
        OfflineAlgorithmWithDefaultOptions,
    };
    use soco::config::Config;
    use soco::convert::{DiscretizableSchedule, RelaxableSchedule};
    use soco::norm::{euclidean, manhattan_scaled, norm_squared};
    use soco::objective::Objective;
    use soco::problem::SmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let t_end = 2;
        let p = SmoothedConvexOptimization {
            d: 2,
            t_end,
            bounds: vec![(0., 2.), (0., 1.)],
            switching_cost: manhattan_scaled(vec![1.5, 1.]),
            hitting_cost: inv_e(t_end),
        };
        p.verify().unwrap();

        let result = co
            .solve_with_default_options(p.clone(), false)
            .unwrap();
        result.verify(p.t_end, &p.bounds.iter().map(|&(_, m)| m).collect()).unwrap();

        assert_eq!(
            result.to_i(),
            Schedule::new(vec![
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1])
            ])
        );
        assert_abs_diff_eq!(
            p.objective_function(&result.to_i().to_f()).unwrap(),
            3.5096, epsilon = 1e-4
        );
    }

    #[test]
    fn _2() {
        let t_end = 100;
        let p = SmoothedConvexOptimization {
            d: 2,
            t_end,
            bounds: vec![(0., 8.), (0., 8.)],
            switching_cost: euclidean(),
            hitting_cost: inv_e(t_end),
        };
        p.verify().unwrap();

        let result = co
            .solve_with_default_options(p.clone(), false)
            .unwrap();
        result.verify(p.t_end, &p.bounds.iter().map(|&(_, m)| m).collect()).unwrap();
    }

    #[test]
    fn _3() {
        let euclidean_ = euclidean();

        let d = 4;
        let t_end = 10;
        let p = SmoothedConvexOptimization {
            d,
            t_end,
            bounds: (0..d)
                .map(|_| {
                    (0., Pcg64::seed_from_u64((d * t_end) as u64).gen_range(1..5) as f64)
                })
                .collect(),
            switching_cost: norm_squared(&euclidean_),
            hitting_cost: inv_e(t_end),
        };
        p.verify().unwrap();

        let result = co
            .solve_with_default_options(p.clone(), false)
            .unwrap();
        result.verify(p.t_end, &p.bounds.iter().map(|&(_, m)| m).collect()).unwrap();
    }
}
