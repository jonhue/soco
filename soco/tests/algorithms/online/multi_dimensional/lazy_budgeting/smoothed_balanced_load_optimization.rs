#[cfg(test)]
mod lb {
    use crate::factories::{constant_simple};
    use num::Float;
    use soco::algorithms::online::multi_dimensional::lazy_budgeting::smoothed_balanced_load_optimization::{lb, Options};
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::{Online, SmoothedBalancedLoadOptimization};
    use soco::schedule::Schedule;

    use crate::init;

    #[test]
    fn _1() {
        init();

        let p = SmoothedBalancedLoadOptimization {
            d: 2,
            t_end: 1,
            bounds: vec![1, 1],
            switching_cost: vec![2., 1.],
            hitting_cost: vec![constant_simple(), constant_simple()],
            load: vec![1, 2, 0, 2, 1],
        };
        let mut o = Online { p, w: 0 };
        o.verify().unwrap();

        let t_end = 5;
        let result = o
            .offline_stream(&lb, t_end, Options::default())
            .unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        let p_ssco = o.p.into_ssco();
        assert!(p_ssco.objective_function(&result.0).unwrap().is_finite());
        assert_eq!(
            result.0,
            Schedule::new(vec![
                Config::new(vec![0, 1]),
                Config::new(vec![0, 1]),
                Config::new(vec![0, 0]),
                Config::new(vec![0, 1]),
                Config::new(vec![0, 1]),
            ])
        );
    }
}
