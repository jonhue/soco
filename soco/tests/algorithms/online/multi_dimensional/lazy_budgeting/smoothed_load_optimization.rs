#[cfg(test)]
mod lb {
    use num::Float;
    use soco::algorithms::online::multi_dimensional::lazy_budgeting::smoothed_load_optimization::{lb, Options};
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::{Online, SmoothedLoadOptimization};
    use soco::schedule::Schedule;

    use crate::init;

    #[test]
    fn _1() {
        init();

        let p = SmoothedLoadOptimization {
            d: 2,
            t_end: 1,
            bounds: vec![1, 1],
            switching_cost: vec![2., 1.],
            hitting_cost: vec![1., 2.],
            load: vec![1, 2, 0, 2, 1],
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 4;
        let result = o
            .offline_stream(&lb, t_end, Options { randomized: false })
            .unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert!(o.p.objective_function(&result.0).unwrap().is_finite());
        assert_eq!(
            result.0,
            Schedule::new(vec![
                Config::new(vec![1, 0]),
                Config::new(vec![1, 1]),
                Config::new(vec![1, 0]),
                Config::new(vec![1, 1]),
            ])
        );
    }

    #[test]
    fn _2() {
        init();

        let p = SmoothedLoadOptimization {
            d: 2,
            t_end: 1,
            bounds: vec![1, 1],
            switching_cost: vec![2., 1.],
            hitting_cost: vec![1., 2.],
            load: vec![1, 2, 0, 2, 1],
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 5;
        let result = o
            .offline_stream(&lb, t_end, Options { randomized: true })
            .unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert!(o.p.objective_function(&result.0).unwrap().is_finite());
        assert_eq!(
            result.0,
            Schedule::new(vec![
                Config::new(vec![1, 0]),
                Config::new(vec![1, 1]),
                Config::new(vec![0, 0]),
                Config::new(vec![1, 1]),
                Config::new(vec![1, 0]),
            ])
        );
    }
}
