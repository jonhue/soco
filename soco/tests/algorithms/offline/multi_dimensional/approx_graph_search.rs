mod approx_graph_search {
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::algorithms::offline::{
        multi_dimensional::approx_graph_search::{
            approx_graph_search, Options,
        },
        OfflineOptions,
    };
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;
    use std::sync::Arc;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end: 2,
            bounds: vec![2, 1],
            switching_cost: vec![1.5, 1.],
            hitting_cost: Arc::new(|t, j| {
                Some(t as f64 * (if j[0] == 0 && j[1] == 0 { 1. } else { 0. }))
            }),
        };
        p.verify().unwrap();

        let result = approx_graph_search(
            &p,
            &Options { gamma: Some(2.) },
            &OfflineOptions { inverted: false },
        )
        .unwrap();
        result.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_result = approx_graph_search(
            &p,
            &Options { gamma: Some(2.) },
            &OfflineOptions { inverted: true },
        )
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
        assert_abs_diff_eq!(result.cost, p.objective_function(&result.xs).unwrap());
    }

    #[test]
    fn _2() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end: 100,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: Arc::new(|t, j| {
                let r: f64 = j
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(k, i)| {
                        Pcg64::seed_from_u64(t as u64 * k as u64)
                            .gen_range(0.0..1.)
                            * i as f64
                    })
                    .sum();
                Some(
                    Pcg64::seed_from_u64(t as u64 * r as u64)
                        .gen_range(0.0..1_000_000.),
                )
            }),
        };
        p.verify().unwrap();

        let result = approx_graph_search(
            &p,
            &Options { gamma: Some(2.) },
            &OfflineOptions { inverted: false },
        )
        .unwrap();
        result.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_result = approx_graph_search(
            &p,
            &Options { gamma: Some(2.) },
            &OfflineOptions { inverted: true },
        )
        .unwrap();
        inv_result.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(result.cost, inv_result.cost);
        assert_abs_diff_eq!(result.cost, p.objective_function(&result.xs).unwrap());
    }

    #[test]
    fn _3() {
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
            hitting_cost: Arc::new(|t, j| {
                let r: f64 = j
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(k, i)| {
                        Pcg64::seed_from_u64(t as u64 * k as u64)
                            .gen_range(0.0..1.)
                            * i as f64
                    })
                    .sum();
                Some(
                    Pcg64::seed_from_u64(t as u64 * r as u64)
                        .gen_range(0.0..1_000_000.),
                )
            }),
        };
        p.verify().unwrap();

        let result = approx_graph_search(
            &p,
            &Options { gamma: Some(2.) },
            &OfflineOptions { inverted: false },
        )
        .unwrap();
        result.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_result = approx_graph_search(
            &p,
            &Options { gamma: Some(2.) },
            &OfflineOptions { inverted: true },
        )
        .unwrap();
        inv_result.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_abs_diff_eq!(result.cost, inv_result.cost, epsilon = 1.);
        assert_abs_diff_eq!(
            result.cost,
            p.objective_function(&result.xs).unwrap(), epsilon = 1.
        );
    }
}
