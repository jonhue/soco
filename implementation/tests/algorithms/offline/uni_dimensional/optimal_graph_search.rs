#[cfg(test)]
mod make_pow_of_2 {
    use crate::factories::constant;
    use crate::init;
    use soco::algorithms::offline::uni_dimensional::optimal_graph_search::make_pow_of_2;
    use soco::config::Config;
    use soco::problem::{Problem, SimplifiedSmoothedConvexOptimization};
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1_000,
            bounds: vec![103],
            switching_cost: vec![1.],
            hitting_cost: constant(),
        };
        p.verify().unwrap();
        let transformed_p = make_pow_of_2(p.clone()).unwrap();
        transformed_p.verify().unwrap();

        assert_eq!(transformed_p.t_end, p.t_end);
        assert_eq!(transformed_p.bounds[0], 128);
        assert_abs_diff_eq!(
            transformed_p.switching_cost[0],
            p.switching_cost[0]
        );

        for t in 1..=transformed_p.t_end {
            for j in 0..=transformed_p.bounds[0] {
                assert_abs_diff_eq!(
                    transformed_p.hit_cost(t, Config::single(j)).cost.raw(),
                    if j <= p.bounds[0] {
                        1.
                    } else {
                        j as f64 * (1. + std::f64::EPSILON)
                    }
                );
            }
        }
    }
}

#[cfg(test)]
mod optimal_graph_search {
    use std::sync::Arc;

    use crate::{
        factories::{penalize_zero, random},
        init,
        utils::hash_map,
    };
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;
    use soco::{algorithms::offline::graph_search::CachedPath, config::Config};
    use soco::{
        algorithms::offline::{
            multi_dimensional::optimal_graph_search::optimal_graph_search as md_optimal_graph_search,
            uni_dimensional::optimal_graph_search::{
                make_pow_of_2, optimal_graph_search, Options,
            },
            OfflineAlgorithm, OfflineOptions,
        },
        problem::Problem,
    };
    use soco::{
        model::{
            data_center::{
                loads::LoadProfile,
                model::{
                    DataCenterModel, DataCenterOfflineInput, JobType, Location,
                    ServerType, Source, DEFAULT_KEY,
                },
                models::{
                    energy_consumption::{
                        EnergyConsumptionModel,
                        SimplifiedLinearEnergyConsumptionModel,
                    },
                    energy_cost::{EnergyCostModel, LinearEnergyCostModel},
                    revenue_loss::{
                        MinimalDetectableDelayRevenueLossModel,
                        RevenueLossModel,
                    },
                    switching_cost::{SwitchingCost, SwitchingCostModel},
                },
                DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
            },
            Model,
        },
        problem::{
            IntegralSimplifiedSmoothedConvexOptimization,
            SimplifiedSmoothedConvexOptimization,
        },
    };

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: penalize_zero(),
        };
        p.verify().unwrap();

        let path = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_path = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(path.xs, inv_path.xs);
        assert_abs_diff_eq!(path.cost, inv_path.cost);
        assert_eq!(
            path.xs,
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
        assert_abs_diff_eq!(path.cost, 1.);
        assert_relative_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw(),
            max_relative = 1e-4
        );
    }

    #[test]
    fn _2() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 100,
            bounds: vec![8],
            switching_cost: vec![1.],
            hitting_cost: random(),
        };
        p.verify().unwrap();

        let path = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();
        let inv_path = optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::inverted())
            .unwrap();
        inv_path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(path.xs, inv_path.xs);
        assert_abs_diff_eq!(path.cost, inv_path.cost);
        assert_relative_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw(),
            max_relative = 1e-4
        );
    }

    #[test]
    fn _3() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 1_000,
            bounds: vec![9],
            switching_cost: vec![1.],
            hitting_cost: random(),
        };
        p.verify().unwrap();

        let CachedPath { path: md_path, .. } = md_optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        md_path.xs.verify(p.t_end, &p.bounds).unwrap();

        let transformed_p = make_pow_of_2(p).unwrap();
        let path = optimal_graph_search
            .solve_with_default_options(
                transformed_p.clone(),
                OfflineOptions::default(),
            )
            .unwrap();
        path.xs
            .verify(transformed_p.t_end, &transformed_p.bounds)
            .unwrap();
        let inv_path = optimal_graph_search
            .solve_with_default_options(
                transformed_p.clone(),
                OfflineOptions::inverted(),
            )
            .unwrap();
        inv_path
            .xs
            .verify(transformed_p.t_end, &transformed_p.bounds)
            .unwrap();

        assert!(path.cost.is_finite());
        assert_abs_diff_eq!(path.cost, md_path.cost);
        assert_eq!(path.xs, inv_path.xs);
        assert_abs_diff_eq!(path.cost, inv_path.cost);
        assert_relative_eq!(
            path.cost,
            transformed_p
                .objective_function(&path.xs)
                .unwrap()
                .cost
                .raw(),
            max_relative = 1e-4
        );
    }

    #[test]
    fn _4() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 2,
            bounds: vec![2],
            switching_cost: vec![1.],
            hitting_cost: penalize_zero(),
        };
        p.verify().unwrap();

        let path = optimal_graph_search
            .solve(p.clone(), Options::new(2), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert_eq!(
            path.xs,
            Schedule::new(vec![Config::single(1), Config::single(1)])
        );
        assert_abs_diff_eq!(path.cost, 0.);
        assert_relative_eq!(
            path.cost,
            p.objective_function_with_default(&path.xs, &Config::single(2))
                .unwrap()
                .cost
                .raw(),
            max_relative = 1e-4
        );
    }

    #[test]
    fn _5() {
        init();

        let loads = vec![LoadProfile::raw(vec![1.])];
        let delta = 10. * 60.;
        let m = 32;
        let model = DataCenterModel::new(
            delta,
            vec![Location {
                key: DEFAULT_KEY.to_string(),
                m: hash_map(&[(DEFAULT_KEY.to_string(), m)]),
            }],
            vec![ServerType::default()],
            vec![Source::default()],
            vec![JobType::default()],
            EnergyConsumptionModel::SimplifiedLinear(hash_map(&[(
                DEFAULT_KEY.to_string(),
                SimplifiedLinearEnergyConsumptionModel { phi_max: 1. },
            )])),
            EnergyCostModel::Linear(hash_map(&[(
                DEFAULT_KEY.to_string(),
                LinearEnergyCostModel {
                    cost: Arc::new(|_| 1.),
                },
            )])),
            RevenueLossModel::MinimalDetectableDelay(hash_map(&[(
                DEFAULT_KEY.to_string(),
                MinimalDetectableDelayRevenueLossModel::default(),
            )])),
            SwitchingCostModel::new(hash_map(&[(
                DEFAULT_KEY.to_string(),
                SwitchingCost {
                    energy_cost: 1.,
                    phi_min: 0.5,
                    phi_max: 1.,
                    epsilon: 1.,
                    delta: 1.,
                    tau: 5.,
                    rho: 5.,
                },
            )])),
        );
        let input = DataCenterOfflineInput { loads };

        let p: IntegralSimplifiedSmoothedConvexOptimization<
            DataCenterModelOutputSuccess,
            DataCenterModelOutputFailure,
        > = model.to(input);
        p.verify().unwrap();

        let CachedPath { path: md_path, .. } = md_optimal_graph_search
            .solve_with_default_options(p.clone(), OfflineOptions::default())
            .unwrap();
        md_path.xs.verify(p.t_end, &p.bounds).unwrap();

        let path = optimal_graph_search
            .solve(p.clone(), Options::default(), OfflineOptions::default())
            .unwrap();
        path.xs.verify(p.t_end, &p.bounds).unwrap();

        assert!(path.cost.is_finite());
        assert_abs_diff_eq!(path.cost, md_path.cost);
        assert_relative_eq!(
            path.cost,
            p.objective_function(&path.xs).unwrap().cost.raw(),
            max_relative = 1e-4
        );
    }
}
