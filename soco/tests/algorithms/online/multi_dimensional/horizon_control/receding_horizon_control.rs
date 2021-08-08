#[cfg(test)]
mod rhc {
    use crate::{factories::inv_e};
    use num::Float;
    use soco::algorithms::online::multi_dimensional::horizon_control::receding_horizon_control::{rhc};
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::{Online, Problem, SimplifiedSmoothedConvexOptimization};
    use soco::schedule::Schedule;
    use crate::init;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end: 1,
            bounds: vec![2., 1.],
            switching_cost: vec![1.5, 1.],
            hitting_cost: inv_e(),
        };
        let mut o = Online { p: p.clone(), w: 5 };
        o.verify().unwrap();

        let t_end = 10;
        let result = o.offline_stream(&rhc, t_end, Default::default()).unwrap();
        result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

        assert!(p.objective_function(&result.0).unwrap().cost.is_finite());
        assert_eq!(
            result.0.to_i(),
            Schedule::new(vec![
                Config::new(vec![1, 1]),
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1])
            ])
        );
    }
}
