#![allow(clippy::float_cmp)]

mod opt_backward {
    use std::sync::Arc;

    use soco::algorithms::offline::opt_backward::opt_backward;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::HomProblem;
    use soco::verifiers::VerifiableSchedule;

    #[test]
    fn _1() {
        let p = HomProblem {
            m: 2,
            t_end: 2,
            f: Arc::new(|t, j| {
                Some(t as f64 * (if j == 0. { 1. } else { 0. }))
            }),
            beta: 1.,
        };
        p.verify().unwrap();

        let result = opt_backward(&p).unwrap();
        result.verify(p.m, p.t_end).unwrap();

        assert_eq!(result.to_i(), vec![1, 1]);
    }
}
