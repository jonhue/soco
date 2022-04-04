#[cfg(test)]
mod probabilistic {
    use crate::factories::parabola;
    use crate::init;
    use soco::algorithms::online::uni_dimensional::probabilistic::Memory;
    use soco::algorithms::online::uni_dimensional::randomized::{
        randomized, Relaxation,
    };
    use soco::problem::{Online, SimplifiedSmoothedConvexOptimization};

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: parabola(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o
            .stream(&randomized, |_, _| false, Relaxation::<Memory>::default())
            .unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();
    }

    #[test]
    fn _2() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: parabola(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o
            .offline_stream(&randomized, t_end, Relaxation::<Memory>::default())
            .unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();
    }
}

#[cfg(test)]
mod randomly_biased_greedy {
    use crate::factories::parabola;
    use crate::init;
    use soco::algorithms::online::uni_dimensional::randomized::{
        randomized, Relaxation,
    };
    use soco::algorithms::online::uni_dimensional::randomly_biased_greedy::Memory;
    use soco::problem::{Online, SimplifiedSmoothedConvexOptimization};

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: parabola(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o
            .stream(&randomized, |_, _| false, Relaxation::<Memory>::default())
            .unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();
    }

    #[test]
    fn _2() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: parabola(),
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 1;
        let result = o
            .offline_stream(&randomized, t_end, Relaxation::<Memory>::default())
            .unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();
    }
}
