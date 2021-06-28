mod randomized {
    use soco::algorithms::online::uni_dimensional::randomized::randomized;
    use soco::online::Online;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use std::sync::Arc;

    #[test]
    fn _1() {
        let p: SimplifiedSmoothedConvexOptimization<'_, i32> =
            SimplifiedSmoothedConvexOptimization {
                d: 1,
                t_end: 1,
                bounds: vec![2],
                switching_cost: vec![1.],
                hitting_cost: Arc::new(|t, j| {
                    Some(1. / t as f64 * (j[0] as f64).powi(2))
                }),
            };
        let o = Online { p, w: 0 };
        o.verify().unwrap();

        let result = o.to_f().stream(randomized, |_, _| false, &()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();
    }

    #[test]
    fn _2() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: Arc::new(|t, j| {
                Some(1. / t as f64 * (j[0] as f64).powi(2))
            }),
        };
        let o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 2;
        let result = o.to_f().offline_stream(randomized, t_end, &()).unwrap();
        result.0.verify(t_end, &o.p.bounds).unwrap();
    }
}
