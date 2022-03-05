#[cfg(test)]
mod brcp {
    use crate::{factories::{inv_e,moving_parabola}};
    use crate::init;
    use crate::utils::upper_bounds;
    use soco::algorithms::offline::multi_dimensional::convex_optimization::co;
    use soco::algorithms::offline::uni_dimensional::capacity_provisioning::brcp;
    use soco::algorithms::offline::{
        OfflineAlgorithm, OfflineOptions, OfflineResult,
    };
    use soco::config::Config;
    use soco::convert::DiscretizableSchedule;
    use soco::problem::{Problem, SimplifiedSmoothedConvexOptimization};
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2.],
            switching_cost: vec![1.],
            hitting_cost: inv_e(),
        };
        p.verify().unwrap();

        let result = brcp
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap()
            .xs();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(
            result.to_i(),
            Schedule::new(vec![Config::single(2), Config::single(2)])
        );
    }

    #[test]
    fn _2() {
        init();

        let d = 1;
        let t_end = 10;
        let p = SimplifiedSmoothedConvexOptimization {
            d,
            t_end,
            bounds: vec![10.],
            switching_cost: vec![1.],
            hitting_cost: moving_parabola(5),
        };
        p.verify().unwrap();

        let p_sco = p.clone().into_sco();
        p_sco.verify().unwrap();

        let result_brcp = brcp
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap()
            .xs();
        let cost_brcp = p.objective_function(&result_brcp).unwrap().cost.raw();
        result_brcp.verify(p.t_end, &p.bounds).unwrap();

        let result_co = co
            .solve_with_default_options(
                p_sco.clone(),
                OfflineOptions::default(),
            )
            .unwrap()
            .xs();
        let cost_co = p_sco.objective_function(&result_co).unwrap().cost.raw();
        result_co
            .verify(p_sco.t_end, &upper_bounds(&p_sco.bounds))
            .unwrap();

        assert!(cost_brcp.is_finite());
        assert_abs_diff_eq!(cost_brcp, cost_co);

        assert_eq!(
            result_brcp,
            Schedule::new(vec![
                Config::single(1.0),
                Config::single(2.0),
                Config::single(3.0),
                Config::single(3.5),
                Config::single(0.5),
                Config::single(1.0),
                Config::single(2.0),
                Config::single(3.0),
                Config::single(3.5),
                Config::single(0.0)
            ])
        );
        assert!(result_brcp == result_co);
    }
}
