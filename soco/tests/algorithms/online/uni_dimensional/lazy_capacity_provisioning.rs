mod fractional_lcp {
    use soco::algorithms::online::uni_dimensional::lazy_capacity_provisioning::lcp;
    use soco::config::{Config, FractionalConfig};
    use soco::convert::DiscretizableSchedule;
    use soco::cost::CostFn;
    use soco::online::Online;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: CostFn::new(|t, j: FractionalConfig| {
                t as f64 * (if j[0] == 0. { 1. } else { 0. })
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(lcp, |_, _| false, &()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();
        println!("{:?}", result);

        assert_eq!(result.0.to_i(), Schedule::new(vec![Config::single(1)]));
    }

    #[test]
    fn _2() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: CostFn::new(|t, j: FractionalConfig| {
                t as f64 * (if j[0] == 0. { 1. } else { 0. })
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(lcp, t_end, &()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0.to_i(),
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
    }
}

mod integral_lcp {
    use soco::algorithms::online::uni_dimensional::lazy_capacity_provisioning::lcp;
    use soco::config::{Config, IntegralConfig};
    use soco::cost::CostFn;
    use soco::online::Online;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: CostFn::new(|t, j: IntegralConfig| {
                t as f64 * (if j[0] == 0 { 1. } else { 0. })
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(lcp, |_, _| false, &()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0, Schedule::new(vec![Config::single(0)]));
    }

    #[test]
    fn _2() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: CostFn::new(|t, j: IntegralConfig| {
                t as f64 * (if j[0] == 0 { 1. } else { 0. })
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(lcp, |_, _| false, &()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0, Schedule::new(vec![Config::single(0)]));
    }

    #[test]
    fn _3() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: CostFn::new(|t, j: IntegralConfig| {
                t as f64 * (if j[0] == 0 { 1. } else { 0. })
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(lcp, t_end, &()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0,
            Schedule::new(vec![Config::single(0), Config::single(1)])
        );
    }

    #[test]
    fn _4() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![5],
            switching_cost: vec![1.],
            hitting_cost: CostFn::new(|t, j: IntegralConfig| {
                t as f64 * (if j[0] == 0 { 1. } else { 0. })
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(lcp, t_end, &()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0,
            Schedule::new(vec![Config::single(0), Config::single(1)])
        );
    }
}
