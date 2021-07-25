#[cfg(test)]
mod rbg {
    use crate::init;
    use crate::{factories::inv_e, utils::upper_bounds};
    use soco::algorithms::online::uni_dimensional::randomly_biased_greedy::{
        rbg, Options,
    };
    use soco::norm::manhattan;
    use soco::problem::{Online, SmoothedConvexOptimization};

    #[test]
    fn _1() {
        init();

        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![(0., 2.)],
            switching_cost: manhattan(),
            hitting_cost: inv_e(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.stream(&rbg, |_, _| false, Options::default()).unwrap();
        result
            .0
            .verify(o.p.t_end, &upper_bounds(&o.p.bounds))
            .unwrap();
    }

    #[test]
    fn _2() {
        init();

        let p = SmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![(0., 2.)],
            switching_cost: manhattan(),
            hitting_cost: inv_e(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.offline_stream(&rbg, t_end, Options::default()).unwrap();
        result.0.verify(t_end, &upper_bounds(&o.p.bounds)).unwrap();
    }
}
