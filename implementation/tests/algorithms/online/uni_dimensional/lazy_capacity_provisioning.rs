#[cfg(test)]
mod fractional_lcp {
    use crate::factories::{inv_e, moving_parabola};
    use crate::init;
    use soco::algorithms::offline::uni_dimensional::capacity_provisioning::brcp;
    use soco::algorithms::offline::{
        OfflineAlgorithm, OfflineOptions, OfflineResult,
    };
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

        println!(
            "{:?} >= {:?}",
            o.p.objective_function(&result.0).unwrap().cost.raw(),
            o.p.objective_function(&brcp_xs).unwrap().cost.raw()
        );

        let bounds = result.1.clone().unwrap().bounds;
        for t in 0..t_end as usize {
            println!(
                "{:?} = {:?} <= {:?} <= {:?} = {:?}",
                bounds[t].lower,
                brcp_bounds[t].lower,
                brcp_xs[t][0],
                brcp_bounds[t].upper,
                bounds[t].upper
            );
        }

        for t in 0..bounds.len() {
            assert_eq!(brcp_bounds[t].lower, bounds[t].lower);
            assert_eq!(brcp_bounds[t].upper, bounds[t].upper);

            assert!(brcp_xs[t][0] >= bounds[t].lower);
            assert!(brcp_xs[t][0] <= bounds[t].upper);
        }

        assert!(false)
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
            .unwrap()
            .xs();
        brcp_result.verify(t_end, &o.p.bounds).unwrap();

        println!(
            "{:?} >= {:?}",
            o.p.objective_function(&result.0).unwrap().cost.raw(),
            o.p.objective_function(&brcp_result).unwrap().cost.raw()
        );

        let bounds = result.1.clone().unwrap().bounds;
        for t in 0..t_end as usize {
            println!(
                "{:?} <= {:?} <= {:?}",
                bounds[t].lower, brcp_result[t][0], bounds[t].upper
            );
        }

        for t in 0..bounds.len() {
            assert!(brcp_result[t][0] >= bounds[t].lower);
            assert!(brcp_result[t][0] <= bounds[t].upper);
        }

        assert!(false)
    }
}

#[cfg(test)]
mod integral_lcp {
    use crate::factories::penalize_zero;
    use crate::init;
    use soco::algorithms::online::uni_dimensional::lazy_capacity_provisioning::lcp;
    use soco::config::Config;
    use soco::problem::{Online, SimplifiedSmoothedConvexOptimization};
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
}
