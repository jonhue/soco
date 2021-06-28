mod probabilistic {
    use soco::algorithms::online::uni_dimensional::probabilistic::probabilistic;
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::online::Online;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use std::sync::Arc;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(1. / t as f64 * (j[0] as f64).powi(2))
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(probabilistic, |_, _| false, &()).unwrap();
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
            hitting_cost: Arc::new(|t: i32, j: Config<f64>| {
                Some(t as f64 * (j[0] as f64).powi(2))
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(probabilistic, t_end, &()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0.to_i(),
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
    }
}
