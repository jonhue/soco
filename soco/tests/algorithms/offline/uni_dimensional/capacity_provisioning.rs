#![allow(clippy::float_cmp)]

mod bcp {
    use std::sync::Arc;

    use soco::algorithms::offline::uni_dimensional::capacity_provisioning::bcp;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::SmoothedConvexOptimization;
    use soco::verifiers::{VerifiableProblem, VerifiableSchedule};

    #[test]
    fn _1() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(t as f64 * (if j[0] == 0. { 1. } else { 0. }))
            }),
        };
        p.verify().unwrap();

        let result = bcp(&p).unwrap();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(result.to_i(), vec![vec![1], vec![1]]);
    }
}

mod fcp {
    use std::sync::Arc;

    use soco::algorithms::offline::uni_dimensional::capacity_provisioning::fcp;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::SmoothedConvexOptimization;
    use soco::verifiers::{VerifiableProblem, VerifiableSchedule};

    #[test]
    fn _1() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(t as f64 * (if j[0] == 0. { 1. } else { 0. }))
            }),
        };
        p.verify().unwrap();

        let result = fcp(&p).unwrap();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(result.to_i(), vec![vec![1], vec![1]]);
    }
}
