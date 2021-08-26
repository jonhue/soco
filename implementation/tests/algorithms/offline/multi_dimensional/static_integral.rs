#[cfg(test)]
mod static_integral {
    use crate::factories::penalize_zero;
    use crate::init;
    use crate::utils::upper_bounds;
    use num::Float;
    use soco::algorithms::offline::multi_dimensional::static_integral::static_integral;
    use soco::algorithms::offline::{
        OfflineAlgorithm, OfflineOptions, OfflineResult,
    };
    use soco::config::IntegralConfig;
    use soco::distance::manhattan_scaled;
    use soco::problem::{Problem, SmoothedConvexOptimization};
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

        let p = SmoothedConvexOptimization {
            d: 5,
            t_end: 100,
            bounds: vec![(0, 2), (0, 1), (0, 5), (0, 3), (0, 2)],
            switching_cost: manhattan_scaled(vec![1.5, 1., 4., 2., 1.7]),
            hitting_cost: penalize_zero(),
        };
        p.verify().unwrap();

        let result = static_integral
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap()
            .xs();
        result.verify(p.t_end, &upper_bounds(&p.bounds)).unwrap();

        assert!(p.objective_function(&result).unwrap().cost.is_finite());
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
