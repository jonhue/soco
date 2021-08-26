#[cfg(test)]
mod dual {
    use crate::{factories::inv_e, utils::upper_bounds};
    use num::Float;
    use soco::algorithms::online::multi_dimensional::online_balanced_descent::dual::{dobd, Options};
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::distance::{euclidean};
    use soco::problem::{Online, Problem, SmoothedConvexOptimization};
    use soco::schedule::Schedule;
    use crate::init;

    #[test]
    fn _1() {
        init();

        let p = SmoothedConvexOptimization {
            d: 2,
            t_end: 1,
            bounds: vec![(0., 2.), (0., 1.)],
            switching_cost: euclidean(),
            hitting_cost: inv_e(),
        };
        let mut o = Online { p: p.clone(), w: 0 };
        o.verify().unwrap();

        let t_end = 5;
        let result = o
            .offline_stream(&dobd, t_end, Options::euclidean_squared(1.))
            .unwrap();
        result
            .0
            .verify(o.p.t_end, &upper_bounds(&o.p.bounds))
            .unwrap();

        assert!(p.objective_function(&result.0).unwrap().cost.is_finite());
        assert_eq!(
            result.0.to_i(),
            Schedule::new(vec![
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1]),
                Config::new(vec![2, 1])
            ])
        );
    }
}
