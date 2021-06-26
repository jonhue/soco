mod bcp {
    use soco::algorithms::offline::uni_dimensional::capacity_provisioning::brcp;
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;
    use std::sync::Arc;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(t as f64 * (if j[0] == 0. { 1. } else { 0. }))
            }),
        };
        p.verify().unwrap();

        let result = brcp(&p).unwrap();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(
            result.to_i(),
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
    }
}
