#[cfg(test)]
mod static_integral {
    use crate::factories::penalize_zero;
    use crate::init;
    use num::Float;
    use soco::algorithms::offline::{
        multi_dimensional::static_integral::static_integral,
    };
    use soco::algorithms::offline::{OfflineAlgorithm, OfflineOptions, OfflineResult};
    use soco::config::IntegralConfig;
    use soco::objective::Objective;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 5,
            t_end: 100,
            bounds: vec![2, 1, 5, 3, 2],
            switching_cost: vec![1.5, 1., 4., 2., 1.7],
            hitting_cost: penalize_zero(),
        };
        p.verify().unwrap();

        let result = static_integral
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap()
            .xs();
        result.verify(p.t_end, &p.bounds).unwrap();

        assert!(p.objective_function(&result).unwrap().is_finite());
        assert!(
            p.total_movement(&result, false).unwrap()
                == p.movement(
                    IntegralConfig::repeat(0, p.d),
                    result[0].clone(),
                    false
                )
        );
    }
}
