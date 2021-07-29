use crate::{init, utils::hash_map};
use soco::{algorithms::offline::{
        multi_dimensional::optimal_graph_search::{
            optimal_graph_search, Options,
        },
        OfflineOptions,
    }, model::{
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
                    MinimalDetectableDelayRevenueLossModel, RevenueLossModel,
                },
                switching_cost::{SwitchingCost, SwitchingCostModel},
            },
            DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
        },
        Model,
    }, problem::{IntegralSimplifiedSmoothedConvexOptimization, Online, Problem}, streaming::offline};
use std::sync::Arc;

#[test]
fn solve() {
    init();

    let loads = vec![
        LoadProfile::raw(vec![10.]),
        LoadProfile::raw(vec![8.]),
        LoadProfile::raw(vec![2.]),
        LoadProfile::raw(vec![5.]),
        LoadProfile::raw(vec![11.]),
        LoadProfile::raw(vec![12.]),
        LoadProfile::raw(vec![11.8]),
        LoadProfile::raw(vec![9.7]),
        LoadProfile::raw(vec![3.4]),
        LoadProfile::raw(vec![0.5]),
        LoadProfile::raw(vec![0.3]),
        LoadProfile::raw(vec![0.4]),
        LoadProfile::raw(vec![0.2]),
        LoadProfile::raw(vec![1.]),
    ];
    let t_end = loads.len() as i32;
    let delta = 1.;
    let m = 100;
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

    let (xs, cost, _) = offline::solve(
        &model,
        &optimal_graph_search,
        Options::default(),
        OfflineOptions::default(),
        input.clone(),
    )
    .unwrap();
    xs.verify(t_end, &vec![m]).unwrap();

    let online_inputs = input.into_online();
    let mut online = Online::<IntegralSimplifiedSmoothedConvexOptimization<DataCenterModelOutputSuccess, DataCenterModelOutputFailure>> {
        p: model.to(DataCenterOfflineInput::default()),
        w: 0,
    };
    for online_input in online_inputs {
        model.update(&mut online, online_input);
    }
    let online_cost = online.p.objective_function(&xs).unwrap();
    assert!(cost == online_cost);
}
