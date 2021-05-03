mod lcp {
    use std::sync::Arc;

    use soco::algorithms::lcp::lcp::lcp;
    use soco::convert::DiscretizableSchedule;
    use soco::online::Online;
    use soco::problem::Problem;
    use soco::verifiers::VerifiableSchedule;

    #[test]
    fn _1() {
        let p = Problem {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_costs: vec![1.],
            f: Arc::new(|t, j| {
                Some(t as f64 * (if j[0] == 0. { 1. } else { 0. }))
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(lcp, |_, _, _| false).unwrap();
        result.0.verify(o.p.t_end, &o.p.switching_costs).unwrap();

        assert_eq!(result.0.to_i(), vec![vec![1]]);
    }

    #[test]
    fn _2() {
        let p = Problem {
            d: 1,
            t_end: 1,
            bounds: vec![2.],
            switching_costs: vec![1.],
            f: Arc::new(|t, j| {
                Some(t as f64 * (if j[0] == 0. { 1. } else { 0. }))
            }),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(lcp, t_end).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();

        assert_eq!(result.0.to_i(), vec![vec![1], vec![1]]);
    }
}
