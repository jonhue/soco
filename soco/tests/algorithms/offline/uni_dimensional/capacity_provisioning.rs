mod bcp {
    use crate::factories::inv_e;
    use soco::algorithms::offline::{uni_dimensional::capacity_provisioning::brcp, OfflineAlgorithmWithDefaultOptions};
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::cost::CostFn;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2.],
            switching_cost: vec![1.5],
            hitting_cost: CostFn::new(inv_e),
        };
        p.verify().unwrap();

        let result = brcp.solve(p.clone(), false).unwrap();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(
            result.to_i(),
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
    }
}
