#[cfg(test)]
mod fractional_lcp {
    use crate::factories::inv_e;
    use crate::init;
    use soco::algorithms::online::uni_dimensional::lazy_capacity_provisioning::lcp;
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::{Online, SimplifiedSmoothedConvexOptimization};
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