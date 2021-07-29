#[cfg(test)]
mod approx_graph_search {
    use crate::{
        factories::{penalize_zero, random},
        init,
    };
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::{
        algorithms::offline::{
            graph_search::CachedPath,
            multi_dimensional::approx_graph_search::{
                approx_graph_search, Options,
            },
            OfflineAlgorithm, OfflineOptions,
        },
        config::Config,
        problem::{Problem, SimplifiedSmoothedConvexOptimization},
        schedule::Schedule,
        verifiers::VerifiableProblem,
    };

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

        let CachedPath { path, .. } = approx_graph_search
            .solve(p.clone(), Options::new(2.), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let CachedPath { path: inv_path, .. } = approx_graph_search
            .solve(p.clone(), Options::new(2.), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(path.xs, inv_path.xs);
        assert_eq!(path.cost, inv_path.cost);
        assert_eq!(
            path.xs,
            Schedule::new(vec![
                Config::new(vec![0, 1]),
                Config::new(vec![0, 1])
            ])
        );
        assert_eq!(path.cost, 1.);
        assert_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw()
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

        let CachedPath { path, .. } = approx_graph_search
            .solve(p.clone(), Options::new(2.), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let CachedPath { path: inv_path, .. } = approx_graph_search
            .solve(p.clone(), Options::new(2.), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(path.cost, inv_path.cost);
        assert_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw()
        );
    }

    #[test]
    fn _3() {
        init();

        let d = 4;
        let t_end = 100;
        let p = SimplifiedSmoothedConvexOptimization {
            d,
            t_end,
            bounds: (0..d)
                .map(|_| {
                    Pcg64::seed_from_u64((d * t_end) as u64).gen_range(1..10)
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

        let CachedPath { path, .. } = approx_graph_search
            .solve(p.clone(), Options::new(2.), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let CachedPath { path: inv_path, .. } = approx_graph_search
            .solve(p.clone(), Options::new(2.), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_relative_eq!(path.cost, inv_path.cost, max_relative = 1e-4);
        assert_relative_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw(),
            max_relative = 1e-4
        );
    }

    #[test]
    fn _4() {
        init();

        let mut p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end: 25,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: random(),
        };
        p.verify().unwrap();

        let CachedPath { path, cache } = approx_graph_search
            .solve(p.clone(), Options::new(2.), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw()
        );

        p.t_end = 100;
        p.verify().unwrap();

        let CachedPath { path, .. } = approx_graph_search
            .solve(
                p.clone(),
                Options {
                    cache: Some(cache),
                    gamma: 2.,
                },
                OfflineOptions::default(),
            )
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw()
        );
    }
}
