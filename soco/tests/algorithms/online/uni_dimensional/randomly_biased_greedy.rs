mod rbg {
    use crate::factories::inv_e;
    use soco::algorithms::online::uni_dimensional::randomly_biased_greedy::{
        rbg, Options,
    };
    use soco::cost::CostFn;
    use soco::norm::manhattan;
    use soco::problem::{Online, SmoothedConvexOptimization};

    #[test]
    fn _1() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![(0., 2.)],
            switching_cost: manhattan(),
            hitting_cost: CostFn::new(inv_e),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(rbg, |_, _| false, Options::default()).unwrap();
        result
            .0
            .verify(
                o.p.t_end,
                &o.p.bounds.into_iter().map(|(_, m)| m).collect(),
            )
            .unwrap();
    }

    #[test]
    fn _2() {
        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![(0., 2.)],
            switching_cost: manhattan(),
            hitting_cost: CostFn::new(inv_e),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(rbg, t_end, Options::default()).unwrap();
        result
            .0
            .verify(t_end, &o.p.bounds.into_iter().map(|(_, m)| m).collect())
            .unwrap();
    }
}
