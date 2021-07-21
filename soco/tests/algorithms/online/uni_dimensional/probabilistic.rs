mod probabilistic {
    use crate::factories::parabola;
    use soco::algorithms::online::uni_dimensional::probabilistic::{
        probabilistic, Options,
    };
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::{Online, SimplifiedSmoothedConvexOptimization};
    use soco::schedule::Schedule;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: parabola(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o
            .stream(&probabilistic, |_, _| false, Options::default())
            .unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0.into_i(), Schedule::new(vec![Config::single(1)]));
    }

    #[test]
    fn _2() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: parabola(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o
            .offline_stream(&probabilistic, t_end, Options::default())
            .unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(
            result.0.into_i(),
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
    }
}
