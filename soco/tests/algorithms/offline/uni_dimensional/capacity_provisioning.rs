mod bcp {
    use crate::factories::inv_e;
    use soco::algorithms::offline::{
        uni_dimensional::capacity_provisioning::brcp,
        OfflineAlgorithmWithDefaultOptions,
    };
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let t_end = 2;
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: inv_e(t_end),
        };
        p.verify().unwrap();

        let result = brcp.solve_with_default_options(p.clone(), false).unwrap();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(
            result.to_i(),
            Schedule::new(vec![Config::single(2), Config::single(2)])
        );
    }
}
