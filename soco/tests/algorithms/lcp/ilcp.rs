mod ilcp {
    use std::sync::Arc;

    use soco::algorithms::lcp::ilcp::ilcp;
    use soco::online::Online;
    use soco::problem::HomProblem;
    use soco::verifiers::VerifiableSchedule;

    #[test]
    fn _1() {
        let p = HomProblem {
            m: 2,
            t_end: 1,
            f: Arc::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
            beta: 1.,
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(ilcp, |_, _, _| false).unwrap();
        result.0.verify(o.p.m, o.p.t_end).unwrap();

        assert_eq!(result.0, vec![0]);
    }

    #[test]
    fn _2() {
        let p = HomProblem {
            m: 2,
            t_end: 1,
            f: Arc::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
            beta: 1.,
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(ilcp, t_end).unwrap();
        result.0.verify(o.p.m, t_end).unwrap();

        assert_eq!(result.0, vec![0, 1]);
    }
}
