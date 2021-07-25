#[cfg(test)]
mod optimal_graph_search {
    use crate::factories::{penalize_zero, random};
    use crate::init;
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::algorithms::offline::graph_search::CachedPath;
    use soco::algorithms::offline::multi_dimensional::optimal_graph_search::Options;
    use soco::algorithms::offline::{
        multi_dimensional::optimal_graph_search::optimal_graph_search,
        OfflineAlgorithmWithDefaultOptions,
    };
    use soco::algorithms::offline::{OfflineAlgorithm, OfflineOptions};
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end: 2,
            bounds: vec![2, 1],
            switching_cost: vec![1.5, 1.],
            hitting_cost: penalize_zero(),
        };
        p.verify().unwrap();

        let CachedPath { path, .. } = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let CachedPath { path: inv_path, .. } = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(path.xs, inv_path.xs);
        assert_abs_diff_eq!(path.cost, inv_path.cost);
        assert_eq!(
            path.xs,
            Schedule::new(vec![
                Config::new(vec![0, 1]),
                Config::new(vec![0, 1])
            ])
        );
        assert_abs_diff_eq!(path.cost, 1.);
        assert_abs_diff_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().raw(),
            epsilon = 1.
        );
    }

    #[test]
    fn _2() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end: 100,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: random(),
        };
        p.verify().unwrap();

        let CachedPath { path, .. } = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let CachedPath { path: inv_path, .. } = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(path.cost, inv_path.cost);
        assert_abs_diff_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().raw(),
            epsilon = 1.
        );
    }

    #[test]
    fn _3() {
        init();

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
            hitting_cost: random(),
        };
        p.verify().unwrap();

        let CachedPath { path, .. } = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let CachedPath { path: inv_path, .. } = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(path.cost, inv_path.cost, epsilon = 1.);
        assert_abs_diff_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().raw(),
            epsilon = 1.
        );
    }

    #[test]
    fn _4() {
        init();

        let d = 4;
        let t_end = 5;
        let mut p = SimplifiedSmoothedConvexOptimization {
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
            hitting_cost: random(),
        };
        p.verify().unwrap();

        let CachedPath { path, cache } = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().raw(),
            epsilon = 1.
        );

        p.t_end = 10;
        p.verify().unwrap();

        let CachedPath { path, .. } = optimal_graph_search
            .solve(
                p.clone(),
                Options { cache: Some(cache) },
                OfflineOptions::default(),
            )
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().raw(),
            epsilon = 1.
        );
    }
}
