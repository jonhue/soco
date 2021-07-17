mod optimal_graph_search {
    use crate::factories::{penalize_zero, random};
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::algorithms::offline::{
        multi_dimensional::optimal_graph_search::optimal_graph_search,
        OfflineAlgorithmWithDefaultOptions,
    };
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let t_end = 2;
        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end,
            bounds: vec![2, 1],
            switching_cost: vec![1.5, 1.],
            hitting_cost: penalize_zero(t_end),
        };
        p.verify().unwrap();

        let result = optimal_graph_search
            .solve_with_default_options(p.clone(), false)
            .unwrap();
        result.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_result = optimal_graph_search
            .solve_with_default_options(p.clone(), true)
            .unwrap();
        inv_result.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(result.xs, inv_result.xs);
        assert_abs_diff_eq!(result.cost, inv_result.cost);
        assert_eq!(
            result.xs,
            Schedule::new(vec![
                Config::new(vec![0, 1]),
                Config::new(vec![0, 1])
            ])
        );
        assert_abs_diff_eq!(result.cost, 1.);
        assert_abs_diff_eq!(
            result.cost,
            p.objective_function(&result.xs).unwrap()
        );
    }

    #[test]
    fn _2() {
        let t_end = 100;
        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: random(t_end),
        };
        p.verify().unwrap();

        let result = optimal_graph_search
            .solve_with_default_options(p.clone(), false)
            .unwrap();
        result.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_result = optimal_graph_search
            .solve_with_default_options(p.clone(), true)
            .unwrap();
        inv_result.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(result.cost, inv_result.cost);
        assert_abs_diff_eq!(
            result.cost,
            p.objective_function(&result.xs).unwrap()
        );
    }

    #[test]
    fn _3() {
        let d = 4;
        let t_end = 10;
        let p = SimplifiedSmoothedConvexOptimization {
            d,
            t_end,
            bounds: (0..d)
                .map(|_| {
                    Pcg64::seed_from_u64((d * t_end) as u64).gen_range(1..5)
                })
                .collect(),
            switching_cost: (0..d)
                .map(|_| {
                    Pcg64::seed_from_u64((d * t_end) as u64).gen_range(1.0..5.)
                })
                .collect(),
            hitting_cost: random(t_end),
        };
        p.verify().unwrap();

        let result = optimal_graph_search
            .solve_with_default_options(p.clone(), false)
            .unwrap();
        result.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_result = optimal_graph_search
            .solve_with_default_options(p.clone(), true)
            .unwrap();
        inv_result.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(result.cost, inv_result.cost, epsilon = 1.);
        assert_abs_diff_eq!(
            result.cost,
            p.objective_function(&result.xs).unwrap(),
            epsilon = 1.
        );
    }
}
