#![allow(clippy::float_cmp)]

mod opt_backward {
    use std::sync::Arc;

    use soco::algorithms::offline::opt::opt_backward;
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

        let result = opt_backward(&p).unwrap();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(result.to_i(), vec![vec![1], vec![1]]);
    }
}

mod opt_forward {
    use std::sync::Arc;

    use soco::algorithms::offline::opt::opt_forward;
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

        let result = opt_forward(&p).unwrap();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(result.to_i(), vec![vec![1], vec![1]]);
    }
}
