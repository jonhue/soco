mod probabilistic {
    use soco::algorithms::online::uni_dimensional::probabilistic::{
        probabilistic, Options,
    };
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
                1. / t as f64 * (j[0] as f64).powi(2)
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o
            .stream(probabilistic, |_, _| false, &Options::default())
            .unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0.to_i(), Schedule::new(vec![Config::single(1)]));
    }

    #[test]
    fn _2() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: CostFn::new(|t: i32, j: FractionalConfig| {
                t as f64 * (j[0] as f64).powi(2)
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o
            .offline_stream(probabilistic, t_end, &Options::default())
            .unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0.to_i(),
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
    }
}
