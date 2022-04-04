#[cfg(test)]
mod fractional_lcp {
    use crate::factories::{inv_e, moving_parabola, parabola};
    use crate::init;
    use soco::algorithms::offline::uni_dimensional::capacity_provisioning::brcp;
    use soco::algorithms::offline::{OfflineAlgorithm, OfflineOptions};
    use soco::algorithms::online::uni_dimensional::lazy_capacity_provisioning::lcp;
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::{
        Online, Problem, SimplifiedSmoothedConvexOptimization,
    };
    use soco::schedule::Schedule;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: inv_e(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(&lcp, |_, _| false, ()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0.to_i(), Schedule::new(vec![Config::single(0)]));
    }

    #[test]
    fn _2() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: inv_e(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(&lcp, t_end, ()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0.to_i(),
            Schedule::new(vec![Config::single(0), Config::single(2)])
        );
    }

    #[test]
    fn _3() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![10_000.],
            switching_cost: vec![1.],
            hitting_cost: inv_e(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 10;
        let result = o.offline_stream(&lcp, t_end, ()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        let brcp_result = brcp
            .solve_with_default_options(o.p.clone(), OfflineOptions::default())
            .unwrap();
        let brcp_xs = brcp_result.xs;
        let brcp_bounds = brcp_result.bounds;
        brcp_xs.verify(t_end, &o.p.bounds).unwrap();

        assert!(
            o.p.objective_function(&result.0).unwrap().cost.raw()
                >= o.p.objective_function(&brcp_xs).unwrap().cost.raw()
        );

        let bounds = result.1.unwrap().bounds;
        for t in 0..bounds.len() {
            assert_eq!(brcp_bounds[t].lower, bounds[t].lower);
            assert_eq!(brcp_bounds[t].upper, bounds[t].upper);

            assert!(brcp_xs[t][0] >= bounds[t].lower);
            assert!(brcp_xs[t][0] <= bounds[t].upper);
        }
    }

    #[test]
    fn _4() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![10.],
            switching_cost: vec![1.],
            hitting_cost: moving_parabola(5),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 10;
        let result = o.offline_stream(&lcp, t_end, ()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        let brcp_result = brcp
            .solve_with_default_options(o.p.clone(), OfflineOptions::default())
            .unwrap();
        let brcp_xs = brcp_result.xs;
        let brcp_bounds = brcp_result.bounds;
        brcp_xs.verify(t_end, &o.p.bounds).unwrap();

        assert!(
            o.p.objective_function(&result.0).unwrap().cost.raw()
                >= o.p.objective_function(&brcp_xs).unwrap().cost.raw()
        );

        // May error due to numerical inaccuracies.
        // assert_eq!(
        //     result.0,
        //     Schedule::new(vec![
        //         Config::single(0.5),
        //         Config::single(1.5),
        //         Config::single(2.5),
        //         Config::single(3.5),
        //         Config::single(0.5),
        //         Config::single(0.5),
        //         Config::single(1.5),
        //         Config::single(2.5),
        //         Config::single(3.5),
        //         Config::single(0.5)
        //     ])
        // );

        let bounds = result.1.unwrap().bounds;
        for t in 0..bounds.len() {
            assert_eq!(brcp_bounds[t].lower, bounds[t].lower);
            assert_eq!(brcp_bounds[t].upper, bounds[t].upper);

            assert!(brcp_xs[t][0] >= bounds[t].lower);
            assert!(brcp_xs[t][0] <= bounds[t].upper);
        }
    }

    /// Selecting optimal $t_start$.
    #[test]
    fn _5() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![10.],
            switching_cost: vec![1.],
            hitting_cost: parabola(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 10;
        let result = o.offline_stream(&lcp, t_end, ()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert!(o
            .p
            .objective_function(&result.0)
            .unwrap()
            .cost
            .raw()
            .is_finite());
    }
}

#[cfg(test)]
mod integral_lcp {
    use crate::factories::{moving_parabola, penalize_zero};
    use crate::init;
    use soco::algorithms::offline::uni_dimensional::optimal_graph_search::optimal_graph_search;
    use soco::algorithms::offline::{OfflineAlgorithm, OfflineOptions};
    use soco::algorithms::online::uni_dimensional::lazy_capacity_provisioning::lcp;
    use soco::config::Config;
    use soco::problem::{
        Online, Problem, SimplifiedSmoothedConvexOptimization,
    };
    use soco::schedule::Schedule;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: penalize_zero(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(&lcp, |_, _| false, ()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0, Schedule::new(vec![Config::single(0)]));
    }

    #[test]
    fn _2() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: penalize_zero(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(&lcp, |_, _| false, ()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0, Schedule::new(vec![Config::single(0)]));
    }

    #[test]
    fn _3() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: penalize_zero(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(&lcp, t_end, ()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0,
            Schedule::new(vec![Config::single(0), Config::single(1)])
        );
    }

    #[test]
    fn _4() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: penalize_zero(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(&lcp, t_end, ()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0,
            Schedule::new(vec![Config::single(0), Config::single(1)])
        );
    }

    #[test]
    fn _5() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![10],
            switching_cost: vec![1.],
            hitting_cost: moving_parabola(5),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 10;
        let result = o.offline_stream(&lcp, t_end, ()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        let opt_result = optimal_graph_search
            .solve_with_default_options(o.p.clone(), OfflineOptions::default())
            .unwrap();
        let opt_xs = opt_result.xs;
        opt_xs.verify(t_end, &o.p.bounds).unwrap();

        assert!(
            o.p.objective_function(&result.0).unwrap().cost.raw()
                >= o.p.objective_function(&opt_xs).unwrap().cost.raw()
        );

        assert_eq!(
            result.0,
            Schedule::new(vec![
                Config::single(0),
                Config::single(1),
                Config::single(2),
                Config::single(3),
                Config::single(0),
                Config::single(0),
                Config::single(1),
                Config::single(2),
                Config::single(3),
                Config::single(0)
            ])
        );

        let bounds = result.1.unwrap().bounds;
        for t in 0..bounds.len() {
            assert!(opt_xs[t][0] >= bounds[t].lower);
            assert!(opt_xs[t][0] <= bounds[t].upper);
        }
    }
}
